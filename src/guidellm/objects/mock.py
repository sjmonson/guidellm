import sys
from importlib.abc import Loader
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType


class Dummy:
    def __getattribute__(self, name):
        return type(self)()

    def __call__(self, *args, **kwargs):
        return type(self)()


class MockModule(ModuleType):
    def __getattr__(self, name):
        # Return a Dummy for any attribute accessed
        return Dummy()


class MockLoader(Loader):
    def create_module(self, spec):
        # Create an instance of MockModule instead of a regular module
        return MockModule(spec.name)

    def exec_module(self, module):
        # No additional setup needed in this case
        pass


# Install the mock module dynamically
mock_name = __name__
spec = spec_from_loader(mock_name, MockLoader())
module = module_from_spec(spec)
sys.modules[mock_name] = module
