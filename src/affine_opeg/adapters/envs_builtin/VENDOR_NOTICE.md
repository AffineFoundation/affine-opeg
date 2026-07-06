# Vendored third-party code

`phybench_eed.py`, `phybench_extended_zss.py`, `phybench_latex_pre_process.py`
are vendored from **github.com/phybench-official/phybench** (`EED/`), used because
the published `phybench` PI-hub wheel ships only `phybench.py` and omits the EED
module it imports.

- **License:** MIT, Copyright (c) 2025 phybench-official.
- `phybench_extended_zss.py` is a modified version of the `zss` package,
  Copyright Tim Henderson and Steve Johnson (also MIT).

MIT permits vendoring/modification/redistribution provided this notice is kept.

```
MIT License

Copyright (c) 2025 phybench-official

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
