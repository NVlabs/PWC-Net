
from torch.utils.ffi import _wrap_function

_corr = imp.load_dynamic('_corr','/usr/local/lib/python2.7/site-packages/correlation_package/_ext/corr/_corr.so')
from _corr import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)

_import_symbols(locals())
