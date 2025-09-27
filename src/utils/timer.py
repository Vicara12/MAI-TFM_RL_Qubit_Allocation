from time import time
from typing import Self, Callable, Dict


class Timer:
  ''' Class for timing stuff.
  '''
  INSTANCES: Dict[str, Self] = {}

  def __init__(self, name: str) -> Self:
    self.total_t = 0
    self.calls = 0
    self.init_t = None
    self.name = name
    self.last_t = None
    if self.name in Timer.INSTANCES.keys():
      raise Exception(f"a timer called {name} already exists")
    Timer.INSTANCES[self.name] = self

  def start(self) -> None:
    ''' Start a new timer.

    If start is called twice without a stop in the middle, the call throws an exception.
    '''
    if self.init_t is not None:
      raise Exception(f"timer.start called twice for timer {self.name}")
    self.init_t = time()
  
  def stop(self) -> float:
    ''' Stop a running timer.

    If there is no running timer the call throws an exception.
    '''
    end = time()
    if self.init_t is None:
      raise Exception(f"timer.stop called before timer.start for timer {self.name}")
    run_time = end - self.init_t
    self.total_t += run_time
    self.last_t = run_time
    self.calls += 1
    self.init_t = None
    return run_time
  
  @property
  def time(self) -> float:
    ''' Returns duration of last measurement.
    '''
    return self.last_t

  @property
  def avg_time(self) -> float:
    ''' Returns the average time over all calls.

    If there has been no call to start and stop this call throws an exception.
    '''
    if self.calls == 0:
      raise Exception(f"empty timer.time called for timer {self.name}")
    return self.total_t/self.calls
  
  @property
  def total_time(self) -> float:
    return self.total_t
  
  @property
  def freq(self) -> float:
    ''' Returns the frequency (1/time) of the calls.
    '''
    return 1/self.time
  
  def reset(self) -> None:
    ''' Reset all parameters.
    '''
    self.total_t = 0
    self.calls = 0
    self.init_t = None
    self.last_t = None
  
  @property
  def timer_decorator(self) -> Callable:
    ''' Time the execution of a function.
    
    Put this decorator over a function in order to measure its execution time. For example:

    @t.timer_decorator
    def foo():
      ...
    
    Or if you don't want to bother with global timers floating around, you can also do:

    @Timer.get("timer_name").timer_decorator
    def foo():
      ...
    
    And then access its contents via Timer.get("timer_name").time just make sure to assign a
    different name to each function.
    '''
    # There is some 5D chess going on here so I'll explain the code. In order to be able to use
    # this as Timer.get(...).timer_decorator this is not a function decorator, but rather a function
    # that returns a function decorator.
    def actual_decorator(func: Callable) -> Callable:
      def wrapper(*args, **kwargs):
        self.start()
        res = func(*args, **kwargs)
        self.stop()
        return res
      return wrapper
    return actual_decorator
  
  @staticmethod
  def get(name: str) -> Self:
    ''' Get a Timer instance by name.
    '''
    if name not in Timer.INSTANCES.keys():
      return Timer(name)
    return Timer.INSTANCES[name]
  
  def __enter__(self):
    self.start()
  
  def __exit__(self, exc_type, exc_value, traceback):
    self.stop()