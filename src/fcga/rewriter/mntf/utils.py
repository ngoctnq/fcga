'''
MNTF-specific utilities
'''

def get_function_signature(source_code: str) -> str:
    """
    Get the function signature from the source code.
    The decompiled code looks like this:

    ```
    
    public void foo(int a, int b)
    {
        int c = a + b;
    }
    ```

    Note the empty line before the function definition, and the Allman-like
    brace style. Personally I would do K&R, though.
    """
    return source_code.strip().split("\n")[0].strip()
