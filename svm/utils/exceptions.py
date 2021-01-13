class BaseException(Exception):
    message = "Something went wrong."
    errors = None
    engine_name = "Exception"

    def __init__(self, message=None, errors=None):
        if message:
            self.message = message

        if errors:
            self.errors = errors

    def __str__(self):
        return (
            f"<{self.engine_name}: {self.__class__.__name__}> {self.message}"
        )
