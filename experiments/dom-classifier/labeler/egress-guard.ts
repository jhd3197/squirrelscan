import { lookup } from "node:dns/promises";
import { connect, createServer, isIP, type Socket } from "node:net";
import { isPrivateHostname } from "./capture.ts";

type Resolve = (hostname: string) => Promise<string[]>;
const resolveAll: Resolve = async (hostname) =>
  (await lookup(hostname, { all: true, verbatim: true })).map((entry) => entry.address);

/** Bytes a proxy request head may take before the client is cut off. */
const MAX_HEAD_BYTES = 16_384;
const RESOLVE_TIMEOUT_MS = 3_000;

/**
 * A loopback-only CONNECT proxy that every replay context sends its traffic
 * through. It resolves the requested host itself, refuses the host when ANY of
 * its addresses is private, and connects to exactly the address it checked, so
 * a name that resolves public for a check and private for the connection
 * (DNS rebinding) cannot reach a local service. Playwright routes loopback
 * through a configured proxy too (`<-loopback>`), so `127.0.0.1` is refused
 * here rather than bypassed.
 *
 * Plain-http requests (absolute-form, not CONNECT) are refused: a proxy
 * connection can carry several of them to different hosts, and replays are of
 * https pages whose plain-http assets the browser blocks as mixed content.
 */
export type EgressGuard = {
  /** `http://127.0.0.1:<port>`, for a Playwright `proxy.server`. */
  server: string;
  /** Hosts refused so far, for an operator's log. */
  refused: string[];
  close(): Promise<void>;
};

export async function startEgressGuard(resolve: Resolve = resolveAll): Promise<EgressGuard> {
  const refused: string[] = [];
  const checked = new Map<string, Promise<string | null>>();
  /** One public address to connect to, or null when the host must be refused. */
  const publicAddress = (host: string) => {
    if (!checked.has(host))
      checked.set(
        host,
        (async () => {
          if (isIP(host)) return isPrivateHostname(host) ? null : host;
          if (isPrivateHostname(host)) return null;
          const timeout = new Promise<string[]>((done) => setTimeout(done, RESOLVE_TIMEOUT_MS, []));
          const addresses = await Promise.race([resolve(host).catch(() => []), timeout]);
          if (!addresses.length || addresses.some((address) => isPrivateHostname(address)))
            return null;
          return addresses[0]!;
        })(),
      );
    return checked.get(host)!;
  };
  const refuse = (client: Socket, status: string, host: string) => {
    refused.push(host);
    client.end(`HTTP/1.1 ${status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n`);
  };
  const open = new Set<Socket>();
  const server = createServer((client) => {
    open.add(client);
    client.on("close", () => open.delete(client));
    let head = Buffer.alloc(0);
    client.on("error", () => client.destroy());
    const onData = async (chunk: Buffer) => {
      head = Buffer.concat([head, chunk]);
      const end = head.indexOf("\r\n\r\n");
      if (end < 0) {
        if (head.length > MAX_HEAD_BYTES) client.destroy();
        return;
      }
      client.off("data", onData);
      client.pause();
      const line = head.subarray(0, head.indexOf("\r\n")).toString("latin1");
      const match = /^CONNECT (\[[0-9a-fA-F:.]+\]|[A-Za-z0-9.-]+):(\d{1,5}) HTTP\/1\.[01]$/.exec(
        line,
      );
      if (!match) return refuse(client, "405 Method Not Allowed", line.split(" ")[1] ?? "?");
      const host = match[1]!.replace(/^\[|\]$/g, "").toLowerCase();
      const port = Number(match[2]);
      const address = port > 0 && port < 65_536 ? await publicAddress(host) : null;
      if (!address) return refuse(client, "403 Forbidden", host);
      const upstream: Socket = connect({ host: address, port }, () => {
        client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
        const rest = head.subarray(end + 4);
        if (rest.length) upstream.write(rest);
        client.pipe(upstream);
        upstream.pipe(client);
        client.resume();
      });
      upstream.on("error", () => client.destroy());
      client.on("close", () => upstream.destroy());
    };
    client.on("data", onData);
  });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  // A shared guard must not keep a finished render process alive.
  server.unref();
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Egress guard did not bind");
  return {
    server: `http://127.0.0.1:${address.port}`,
    refused,
    close: () =>
      new Promise<void>((done) => {
        for (const socket of open) socket.destroy();
        server.close(() => done());
      }),
  };
}
