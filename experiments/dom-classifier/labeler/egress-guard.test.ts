import { afterEach, describe, expect, test } from "bun:test";
import { connect } from "node:net";
import { isPublicAddress, startEgressGuard, type EgressGuard } from "./egress-guard.ts";

const guards: EgressGuard[] = [];
const servers: Array<ReturnType<typeof Bun.serve>> = [];
afterEach(async () => {
  for (const guard of guards.splice(0)) await guard.close();
  for (const server of servers.splice(0)) server.stop(true);
});

/** A local service that counts every connection it is handed. */
function localService() {
  const seen = { hits: 0 };
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    fetch() {
      seen.hits++;
      return new Response("local");
    },
  });
  servers.push(server);
  return { port: server.port, seen };
}

/** Send one raw proxy request head and return the status line the guard answers with. */
function ask(guard: EgressGuard, head: string) {
  const port = Number(new URL(guard.server).port);
  return new Promise<string>((resolve, reject) => {
    const socket = connect({ host: "127.0.0.1", port }, () => socket.write(head));
    let reply = "";
    socket.on("data", (chunk) => {
      reply += chunk.toString("latin1");
      if (reply.includes("\r\n")) {
        socket.destroy();
        resolve(reply.split("\r\n")[0]!);
      }
    });
    socket.on("error", reject);
    socket.on("close", () => resolve(reply.split("\r\n")[0] ?? ""));
  });
}

describe("replay egress guard", () => {
  test("refuses a public-looking name that resolves to loopback (DNS rebinding)", async () => {
    const local = localService();
    const guard = await startEgressGuard(async () => ["127.0.0.1"]);
    guards.push(guard);
    const status = await ask(
      guard,
      `CONNECT rebind.example.test:${local.port} HTTP/1.1\r\nHost: rebind.example.test\r\n\r\n`,
    );
    expect(status).toBe("HTTP/1.1 403 Forbidden");
    expect(local.seen.hits).toBe(0);
    expect(guard.refused).toContain("rebind.example.test");
  });

  test("refuses a host when any of its addresses is private", async () => {
    const guard = await startEgressGuard(async () => ["93.184.215.14", "10.0.0.7"]);
    guards.push(guard);
    expect(await ask(guard, "CONNECT mixed.example.test:443 HTTP/1.1\r\n\r\n")).toBe(
      "HTTP/1.1 403 Forbidden",
    );
  });

  test("refuses loopback and private literals without resolving", async () => {
    const local = localService();
    let lookups = 0;
    const guard = await startEgressGuard(async () => {
      lookups++;
      return ["93.184.215.14"];
    });
    guards.push(guard);
    for (const target of [
      `127.0.0.1:${local.port}`,
      `[::1]:${local.port}`,
      "192.168.1.1:80",
      `localhost:${local.port}`,
    ])
      expect(await ask(guard, `CONNECT ${target} HTTP/1.1\r\n\r\n`)).toBe("HTTP/1.1 403 Forbidden");
    expect(lookups).toBe(0);
    expect(local.seen.hits).toBe(0);
  });

  test("classifies every non-public spelling, including unspecified and mapped forms", () => {
    for (const address of [
      "127.0.0.1",
      "0.0.0.0",
      "10.1.2.3",
      "100.64.0.1",
      "169.254.169.254",
      "172.20.0.1",
      "192.168.0.10",
      "224.0.0.1",
      "::",
      "::1",
      "0:0:0:0:0:0:0:1",
      "::ffff:127.0.0.1",
      "::ffff:7f00:1",
      "fd00::1",
      "fe80::1",
      "ff02::1",
      "not-an-ip",
    ])
      expect(isPublicAddress(address)).toBe(false);
    for (const address of ["93.184.215.14", "2606:4700::1111", "::ffff:8.8.8.8"])
      expect(isPublicAddress(address)).toBe(true);
  });

  test("refuses the unspecified address and a trailing-dot local name", async () => {
    const local = localService();
    const guard = await startEgressGuard(async () => ["93.184.215.14"]);
    guards.push(guard);
    for (const target of [
      `[::]:${local.port}`,
      `0.0.0.0:${local.port}`,
      `localhost.:${local.port}`,
    ])
      expect(await ask(guard, `CONNECT ${target} HTTP/1.1\r\n\r\n`)).toBe("HTTP/1.1 403 Forbidden");
    expect(local.seen.hits).toBe(0);
  });

  test("refuses plain-http proxy requests and unresolvable hosts", async () => {
    const local = localService();
    const guard = await startEgressGuard(async () => {
      throw new Error("NXDOMAIN");
    });
    guards.push(guard);
    expect(
      await ask(guard, `GET http://127.0.0.1:${local.port}/ HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n`),
    ).toBe("HTTP/1.1 405 Method Not Allowed");
    expect(await ask(guard, "CONNECT nowhere.example.test:443 HTTP/1.1\r\n\r\n")).toBe(
      "HTTP/1.1 403 Forbidden",
    );
    expect(local.seen.hits).toBe(0);
  });
});
