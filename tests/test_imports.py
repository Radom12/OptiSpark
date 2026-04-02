import optispark

def test_optispark_import():
    assert optispark.__version__ is not None or hasattr(optispark, '__name__')
