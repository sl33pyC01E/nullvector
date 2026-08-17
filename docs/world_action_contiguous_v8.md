# Contiguous cellular action corpus v8

The earlier cellular action corpus retained only isolated action/settle pairs. V8 binds the same six source worlds and frozen world encoder but preserves all 396 consecutive frames per world.

The first four worlds train, the fifth validates, and the sixth remains untouched. This corpus supports truncated recurrent training and honest multi-step rollout gates without duplicating the large raw RGB and cellular teacher archives.
