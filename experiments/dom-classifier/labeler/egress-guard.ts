import { lookup } from "node:dns/promises";
import { BlockList, connect, createServer, isIP, type Socket } from "node:net";

type Resolve = (hostname: string) => Promise<string[]>;
const resolveAll: Resolve = async (hostname) =>
  (await lookup(hostname, { all: true, verbatim: true })).map((entry) => entry.address);

/** Bytes a proxy request head may take before the client is cut off. */
const MAX_HEAD_BYTES = 16_384;
const RESOLVE_TIMEOUT_MS = 3_000;
/** A tunnel idle this long is closed; a replay never needs one longer. */
const IDLE_TIMEOUT_MS = 60_000;
const MAX_CONNECTIONS = 256;
/** Host decisions are cached briefly in a bounded map; refusals are logged up to a cap. */
const CACHE_TTL_MS = 10 * 60_000;
const MAX_CACHED_HOSTS = 2_000;
const MAX_REFUSALS_LOGGED = 1_000;

/**
 * Every address range a replayed page must not reach: unspecified, loopback,
 * private, shared, link-local, special-use, benchmarking, multicast and
 * reserved, for both families. `BlockList` normalises expanded and
 * IPv4-mapped IPv6 spellings, so `::`, `0:0:0:0:0:0:0:1` and `::ffff:7f00:1`
 * are caught as well.
 */
const NON_PUBLIC = new BlockList();
for (const [network, prefix] of [
  ["0.0.0.0", 8],
  ["10.0.0.0", 8],
  ["100.64.0.0", 10],
  ["127.0.0.0", 8],
  ["169.254.0.0", 16],
  ["172.16.0.0", 12],
  ["192.0.0.0", 24],
  ["192.168.0.0", 16],
  ["198.18.0.0", 15],
  ["224.0.0.0", 3],
] as const)
  NON_PUBLIC.addSubnet(network, prefix, "ipv4");
for (const [network, prefix] of [
  ["::", 128],
  ["::1", 128],
  ["64:ff9b::", 96],
  ["100::", 64],
  ["fc00::", 7],
  ["fe80::", 10],
  ["ff00::", 8],
] as const)
  NON_PUBLIC.addSubnet(network, prefix, "ipv6");

/** True only for a literal IP address outside every non-public range. */
export function isPublicAddress(address: string) {
  const family = isIP(address);
  if (!family) return false;
  return !NON_PUBLIC.check(address, family === 4 ? "ipv4" : "ipv6");
}

/**
 * A loopback-only CONNECT proxy that every replay context sends its traffic
 * through. It resolves the requested host itself, refuses the host when ANY of
 * its addresses is not public, and connects to exactly the address it checked,
 * so a name that resolves public for a check and private for the connection
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
  /** Hosts refused so far (capped), for an operator's log. */
  refused: string[];
  close(): Promise<void>;
};

export async function startEgressGuard(resolve: Resolve = resolveAll): Promise<EgressGuard> {
  const refused: string[] = [];
  const checked = new Map<string, { at: number; address: Promise<string | null> }>();
  /** One public address to connect to, or null when the host must be refused. */
  const publicAddress = (host: string) => {
    const cached = checked.get(host);
    if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.address;
    if (checked.size >= MAX_CACHED_HOSTS) checked.clear();
    const address = (async () => {
      if (isIP(host)) return isPublicAddress(host) ? host : null;
      if (host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local"))
        return null;
      const timeout = new Promise<string[]>((done) => setTimeout(done, RESOLVE_TIMEOUT_MS, []));
      const addresses = await Promise.race([resolve(host).catch(() => []), timeout]);
      if (!addresses.length || !addresses.every(isPublicAddress)) return null;
      return addresses[0]!;
    })();
    checked.set(host, { at: Date.now(), address });
    return address;
  };
  const refuse = (client: Socket, status: string, host: string) => {
    if (refused.length < MAX_REFUSALS_LOGGED) refused.push(host);
    client.end(`HTTP/1.1 ${status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n`);
  };
  const open = new Set<Socket>();
  const track = (socket: Socket) => {
    open.add(socket);
    socket.on("close", () => open.delete(socket));
    socket.setTimeout(IDLE_TIMEOUT_MS, () => socket.destroy());
  };
  let closed = false;
  const server = createServer((client) => {
    track(client);
    client.on("error", () => client.destroy());
    let head = Buffer.alloc(0);
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
      const host = match[1]!
        .replace(/^\[|\]$/g, "")
        .replace(/\.$/, "")
        .toLowerCase();
      const port = Number(match[2]);
      const address = port > 0 && port < 65_536 ? await publicAddress(host) : null;
      // The client may have gone, or the guard closed, while the host resolved.
      if (client.destroyed || closed) return client.destroy();
      if (!address) return refuse(client, "403 Forbidden", host);
      const upstream: Socket = connect({ host: address, port }, () => {
        client.write("HTTP/1.1 200 Connection Established\r\n\r\n");
        const rest = head.subarray(end + 4);
        if (rest.length) upstream.write(rest);
        client.pipe(upstream);
        upstream.pipe(client);
        client.resume();
      });
      track(upstream);
      upstream.on("error", () => client.destroy());
      upstream.on("close", () => client.destroy());
      client.on("close", () => upstream.destroy());
    };
    client.on("data", onData);
  });
  server.maxConnections = MAX_CONNECTIONS;
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
        closed = true;
        for (const socket of open) socket.destroy();
        server.close(() => done());
      }),
  };
}
