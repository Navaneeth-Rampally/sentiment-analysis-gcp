import sys, os
from custom_exception import SentimentException

def divide(a, b):
    try:
        if b == 0:
            raise ZeroDivisionError("Division by Zero is not allowed")
        res = a / b
        return res
        
    except ZeroDivisionError as e:
        raise SentimentException(
            error_message=f"Cannot do this division: {str(e)}",
            error_detail=sys.exc_info()
        ) from e
        
    except Exception as e:
        raise SentimentException(
            error_message="Something wrong during calculation",
            error_detail=sys.exc_info()
        ) from e

if __name__ == "__main__":
    try:
        # Note: Your original code printed "50 / 4" but calculated 10/2. 
        # I kept the calculation logic (10/2).
        print("50 / 4 result:", divide(10, 2))

        print("6 / 0 result:", divide(6, 0))
        
    except SentimentException as e:
        # Replaced logger with print so you can see the error
        print(f"Caught exception: {e}")