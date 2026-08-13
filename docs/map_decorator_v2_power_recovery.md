# Map decorator v2 foreground-index power recovery

The v2 foreground index has an additive recovery path for an unpublished staging tree. Recovery never edits, deletes, renames, or publishes the outage tree. It validates every atomically published `counts.json` against the current corpus, v2 contract, source hash, shard identity, input artifact hashes, and ordered sample identities. Valid shard files are copied byte-for-byte into a new unique staging tree; only missing shards are rebuilt.

The new tree is aggregated, validated, and moved to the requested final path with one same-volume directory rename. A failure leaves both the outage evidence and the new recovery staging tree in place.

Audit without writing:

```powershell
python -m forge.map_decorator_production_v2.recovery audit `
  --corpus outputs/map_decorator_corpus_v1 `
  --staging outputs/map_decorator_production_v2/.foreground_index_v2.tmp-<id>
```

Recover and atomically publish:

```powershell
python -m forge.map_decorator_production_v2.recovery recover `
  --corpus outputs/map_decorator_corpus_v1 `
  --source-staging outputs/map_decorator_production_v2/.foreground_index_v2.tmp-<id> `
  --output outputs/map_decorator_production_v2/foreground_index_v2 `
  --workers 2
```

Safety invariants:

- at most two isolated CPU workers;
- at most three attempts per missing shard, with unsigned native exit codes and `0xC0000005` classification in telemetry;
- a 100 GiB free-space floor before copies, worker launches, and publication;
- byte-exact source-tree inventory before and after recovery;
- no overwrite of an existing publication target;
- no CUDA import, calibration, or training;
- final corpus/index validation both before and after atomic publication.
