"""
Builtin interpolation operators for the resample operation.
"""

def FFILL(*args):
  """
  Builtin forward fill interpolation operator. The shorthand notation for this
  operator is `ffill`.

  Interpolates a value at time `t`, using the closest value available before
  time `t` (i.e. interpolation using the value closest in the past).

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.FFILL())

  """
  return ("__builtin__ffill__", [""])

def BFILL(*args):
  """
  Builtin backward fill interpolation operator. The shorthand notation for this
  operator is `bfill`.

  Interpolates a value at time `t`, using the closest value available after
  time `t` (i.e. interpolation using the value closest in the future).

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.BFILL())

  """
  return ("__builtin__bfill__", [""])

def NEAREST(*args):
  """
  Builtin neartest interpolation operator. The shorthand notation for this
  operator is `neartest`.

  Interpolates a value at time `t`, using the value associated with a time
  nearest to `t`.

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.NEAREST())

  """
  return ("__builtin__nearest__", [""])

def ZERO(*args):
  """
  Builtin zero value fill interpolation operator. The shorthand notation for
  this operator is `zero`.

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.ZERO())

  """
  return ("__builtin__zero__", [""])

def NONE(*args):
  """
  Builtin None value fill interpolation operator. The shorthand notation for
  this operator is `none` (or None).

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.NONE())

  """
  return ("__builtin__none__", [""])

def LINEAR(*args):
  """
  Builtin linear interpolation operator. The shorthand notation for this
  operator is `linear`.

  Returns
  --------
  out : Interpolation object.

  Examples
  --------
  >>> ts.resample(timedelta(hour = 1),
                  upsample_method = gl.interpolation.LINEAR())

  """
  return ("__builtin__linear__", [""])
