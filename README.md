TEXT: Numerous tools for text processing
========================================

<img align=right src="https://travis-ci.org/mit-nlp/Text.jl.svg?branch=master" alt="Build Status"/>

This package is a julia implementation of:

1. Text classification based on BoW models (e.g. topic/langauge id)
2. Language ID (training and processing) based on word and character n-grams
3. Lewis's SMART stop list for English
4. tfidf/tfllr text feature normalization
5. ngram feature extractors

Prerequistes
------------

- `Stage`          - Needed for logging and memoization *(Note: requires manual install)*
- `Ollam`          - online learning modules *(Note: requires manual install)*
- `Devectorize`    - macro-based devectorization
- `DataStructures` - for DefaultDict
- `Devectorize`
- `GZip`
- `Iterators`      - for iterator helper functions

Install
-------

This is an experimental package which is not currently registered in
the julia central repository.  You can install via:

```julia
Pkg.clone("https://github.com/saltpork/Stage.jl")
Pkg.clone("https://github.com/mit-nlp/Ollam.jl")
Pkg.clone("https://github.com/mit-nlp/Text.jl")
```

Usage
-----

See `test/runtests.jl` for detailed usage.

License
-------
This package was created for the DARPA XDATA program under an Apache v2 License.

