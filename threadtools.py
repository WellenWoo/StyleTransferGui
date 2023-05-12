# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:16:22 2019
@author: WellenWoo
"""
import wx
import threading
from types import FunctionType, MethodType

__all__ = ["callafter", "synchfunc", "ClassSynchronizer"]


def callafter(func):
    """Decorator to automatically use CallAfter if
    a method is called from a different thread.
    """

    def callafterwrap(*args, **kw):
        #        if wx.Thread_IsMain():   #wx Classic
        if wx.IsMainThread():  # wx Phoenix
            return func(*args, **kw)
        else:
            wx.CallAfter(func, *args, **kw)

    callafterwrap.__name__ = func.__name__
    callafterwrap.__doc__ = func.__doc__
    return callafterwrap


def synchfunc(func):
    """Decorator to synchronize a method call from a worker
    thread to the GUI thread.
    """

    def synchwrap(*args, **kw):
        if wx.IsMainThread():
            return func(*args, **kw)
        else:
            synchobj = Synchronizer(func, args, kw)
            return synchobj.run()

    synchwrap.__name__ = func.__name__
    synchwrap.__doc__ = func.__doc__
    return synchwrap


class Synchronizer(object):
    """Synchronize CallAfter calls"""

    def __init__(self, func, args, kw):
        super(Synchronizer, self).__init__()

        self.func = func
        self.args = args
        self.kw = kw
        self._synch = threading.Semaphore(0)

    def _asynch_wrapper(self):
        """This part runs in main gui thread"""
        try:
            self.result = self.func(*self.args, **self.kw)
        except Exception as msg:
            self.exception = msg

        # Release Semaphore to allow processing back
        # on other thread to resume.    
        self._synch.release()  # 释放线程锁

    def run(self):
        """Call from background thread"""
        # Make sure this is not called from main thread
        # as it will result in deadlock waiting on the
        # Semaphore.
        assert not wx.IsMainThread(), "DeadLock"

        # Make the asynchronous call to the main thread
        # to run the function.
        wx.CallAfter(self._asynch_wrapper)

        # Block on Semaphore release until the function
        # has been processed in the main thread by the
        # UI's event loop.
        self._synch.acquire()  # 阻塞

        # Return result to caller or raise error
        try:
            return self.result
        except AttributeError:
            raise self.exception


class ClassSynchronizer(type):
    """Metaclass to make all methods in a class threadsafe"""

    def __call__(mcs, *args, **kw):
        obj = type.__call__(mcs, *args, **kw)

        # Wrap all methods/functions in the class with
        # the synchfunct decorator.
        for attrname in dir(obj):
            attr = getattr(obj, attrname)
            if type(attr) in (MethodType, FunctionType):
                nfunc = synchfunc(attr)
                setattr(obj, attrname, nfunc)
        return obj
