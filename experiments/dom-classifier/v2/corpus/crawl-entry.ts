#!/usr/bin/env bun
/**
 * Run the squirrel CLI with a bigger content store, without editing the CLI.
 *
 * ## The problem this solves
 *
 * `CONTENT_STORE_MAX_BYTES` is 1 GiB and the store LRU-prunes to 80% of it
 * whenever it goes over. A three-hour crawl therefore evicts its own earlier
 * pages while it runs: the first attempt fetched 70,278 pages and the store
 * kept 26,841 of them, because the project databases know 68,742 content hashes
 * and the blobs behind 41,901 of them were thrown away. The corpus was bounded
 * by a disk constant, not by crawl rate, concurrency or clock — which is why
 * 4 workers, 12 workers and 3 hours all landed in the same place. (Filed as
 * repo#2342; this lane does not change the shipped constant.)
 *
 * ## How it is avoided here
 *
 * `ContentStore`'s constructor already takes a `maxBytes`, but nothing passes
 * one: `getGlobalContentStore()` calls `new ContentStore(desiredPath)`, so the
 * singleton always gets the default. There is no setter for the singleton
 * either. So this file takes the singleton the normal way and RAISES the field
 * on the instance before the crawler has written anything. `private` in
 * TypeScript is a compile-time rule, not a runtime one, and `prune()` and
 * `pruneIfNeeded()` both read `this.maxBytes` at call time.
 *
 * Nothing shipped is modified. This is an entry point that lives in the
 * experiment directory and is used only by `corpus/crawl.ts`.
 *
 * ## The one thing that would silently undo it
 *
 * `getGlobalContentStore()` closes and REBUILDS the singleton when
 * `SQUIRREL_CONTENT_STORE_PATH` is set and disagrees with the store's current
 * path — and a rebuild would restore the 1 GiB default with no error. So the
 * variable is required here and the path is asserted to match after patching,
 * rather than trusted.
 */
import { getGlobalContentStore } from "../../../../apps/cli/src/crawler/storage/content-store.ts";

const target = Number(process.env.SQUIRREL_CONTENT_STORE_MAX_BYTES ?? 0);
if (!Number.isFinite(target) || target <= 0)
  throw new Error("SQUIRREL_CONTENT_STORE_MAX_BYTES must be a positive byte count");

const wanted = process.env.SQUIRREL_CONTENT_STORE_PATH;
if (!wanted)
  throw new Error(
    "SQUIRREL_CONTENT_STORE_PATH must be set explicitly: without it the CLI can " +
      "later set it itself, and a path disagreement silently rebuilds the store at the 1 GiB default",
  );

const store = getGlobalContentStore();
const patched = store as unknown as { maxBytes: number };
const before = patched.maxBytes;
patched.maxBytes = target;

if (store.getPath() !== wanted)
  throw new Error(
    `store opened at ${store.getPath()} but SQUIRREL_CONTENT_STORE_PATH is ${wanted}; ` +
      "the singleton would be rebuilt and the cap lost",
  );
if (patched.maxBytes !== target) throw new Error("maxBytes did not take");

if (process.env.SQUIRREL_CRAWL_ENTRY_VERBOSE)
  console.log(
    JSON.stringify({
      stage: "store-cap",
      path: store.getPath(),
      beforeBytes: before,
      afterBytes: patched.maxBytes,
    }),
  );

// Importing the CLI runs it: cli.ts is three lines and calls run() at module
// scope. By this point the singleton it will ask for is already patched.
await import("../../../../apps/cli/src/cli.ts");
