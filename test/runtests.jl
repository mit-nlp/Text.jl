using Text
using Base.Test, Stage, Ollam, DataStructures

# -------------------------------------------------------------------------------------------------------------------------
# readers
# -------------------------------------------------------------------------------------------------------------------------
if false
l = 0
for i in filelines("data/nus-sms/test.tsv.gz")
  l += 1
end
@expect l == 6717

l = 0
for i in lazy_map(x -> split(strip(x), '\t')[2], filelines("data/nus-sms/test.tsv.gz"))
  l += 1
  if (l == 6717)
    println("6717: ", i)
  end
end
@expect l == 6717
end

# -------------------------------------------------------------------------------------------------------------------------
# tokenizers
# -------------------------------------------------------------------------------------------------------------------------
@expect english_tokenizer("  test  ")     == ["test"]
@expect english_tokenizer("  test.")      == ["test"]
@expect english_tokenizer("\"test.\"")    == ["test"]
@expect english_tokenizer(">test")        == ["test"]
@expect english_tokenizer(">test<")       == ["test"]
@expect english_tokenizer("\$100.0")      == ["--currency--"]
@expect english_tokenizer("\$100")        == ["--currency--"]
@expect english_tokenizer("100.0")        == ["--number--"]
@expect english_tokenizer("https://test") == ["--url--"]
@expect english_tokenizer("ftp://test")   == ["--url--"]
@expect english_tokenizer("http://test")  == ["--url--"]
@expect english_tokenizer("ice-cream")    == ["ice", "cream"]
@expect english_tokenizer("ice  cream")   == ["ice", "cream"]
@expect english_tokenizer("ice--cream")   == ["ice", "cream"]
@expect english_tokenizer("ice_cream")    == ["ice", "cream"]
@expect english_tokenizer("don't")        == ["do", "not"]
@expect english_tokenizer("don't be")     == ["do", "not", "be"]
@expect english_tokenizer("cat's")        == ["cat", "s's"]
@expect english_tokenizer("how'd")        == ["how", "d'd"]

# -------------------------------------------------------------------------------------------------------------------------
# feature extraction
# -------------------------------------------------------------------------------------------------------------------------
# ngrams from arrays
@expect ngrams(["a", "b", "c"], order = 3)                         == ["a", "a b", "a b c"]
@expect ngrams(["a", "b", "c"], order = 3, truncated_start = true) == ["a b c"]

@expect ngrams(["a", "b", "c"], order = 2)                         == ["a", "a b", "b c"]
@expect ngrams(["a", "b", "c"], order = 2, truncated_start = true) == ["a b", "b c"]

@expect ngrams(["a", "b", "c"], order = 1)                         == ["a", "b", "c"]
@expect ngrams(["a", "b", "c"], order = 1, truncated_start = true) == ["a", "b", "c"]

@expect ngrams(["a"], order = 3)                                   == ["a"]
@expect ngrams(["a"], order = 3, truncated_start = true)           == []

# ngrams from strings
@expect ngrams("abc", order = 3)                           == ["a", "ab", "abc"]
@expect ngrams("abc", order = 3, truncated_start = true)   == ["abc"]

@expect ngrams("abc", order = 2)                           == ["a", "ab", "bc"]
@expect ngrams("abc", order = 2, truncated_start = true)   == ["ab", "bc"]

@expect ngrams("abc", order = 1)                           == ["a", "b", "c"]
@expect ngrams("abc", order = 1, truncated_start = true)   == ["a", "b", "c"]

@expect ngrams("a", order = 3)                             == ["a"]
@expect ngrams("ab", order = 3)                            == ["a", "ab"]
@expect ngrams("abcd", order = 3)                          == ["a", "ab", "abc", "bcd"]
@expect ngrams("a", order = 3, truncated_start = true)     == []
@expect ngrams("ab", order = 3, truncated_start = true)    == []
@expect ngrams("abcd", order = 3, truncated_start = true)  == ["abc", "bcd"]

@expect ngrams("是的", order = 1)                          == ["是", "的"]
@expect ngrams("是的", order = 2)                          == ["是", "是的"]
@expect ngrams("是的", order = 3)                          == ["是", "是的"]
@expect ngrams("是的", order = 3, truncated_start = true)  == []

@expect ngrams("陇陇*", order = 1)                         == ["陇", "陇", "*"]
@expect ngrams("陇陇*", order = 2)                         == ["陇", "陇陇", "陇*"]
@expect ngrams("陇陇*", order = 3)                         == ["陇", "陇陇", "陇陇*"]
@expect ngrams("陇陇*", order = 3, truncated_start = true) == ["陇陇*"]

@expect ngrams("", order = 1)                              == []

# ngram iterator
@expect collect(ngram_iterator("abc", order = 3))                           == ["a", "ab", "abc"]
@expect collect(ngram_iterator("abc", order = 3, truncated_start = true))   == ["abc"]

@expect collect(ngram_iterator("abc", order = 2))                           == ["a", "ab", "bc"]
@expect collect(ngram_iterator("abc", order = 2, truncated_start = true))   == ["ab", "bc"]

@expect collect(ngram_iterator("abc", order = 1))                           == ["a", "b", "c"]
@expect collect(ngram_iterator("abc", order = 1, truncated_start = true))   == ["a", "b", "c"]

@expect collect(ngram_iterator("a", order = 3))                             == ["a"]
@expect collect(ngram_iterator("ab", order = 3))                            == ["a", "ab"]
@expect collect(ngram_iterator("abcd", order = 3))                          == ["a", "ab", "abc", "bcd"]
@expect collect(ngram_iterator("a", order = 3, truncated_start = true))     == []
@expect collect(ngram_iterator("ab", order = 3, truncated_start = true))    == []
@expect collect(ngram_iterator("abcd", order = 3, truncated_start = true))  == ["abc", "bcd"]

@expect collect(ngram_iterator("是的", order = 1))                          == ["是", "的"]
@expect collect(ngram_iterator("是的", order = 2))                          == ["是", "是的"]
@expect collect(ngram_iterator("是的", order = 3))                          == ["是", "是的"]
@expect collect(ngram_iterator("是的", order = 3, truncated_start = true))  == []

@expect collect(ngram_iterator("陇陇*", order = 1))                         == ["陇", "陇", "*"]
@expect collect(ngram_iterator("陇陇*", order = 2))                         == ["陇", "陇陇", "陇*"]
@expect collect(ngram_iterator("陇陇*", order = 3))                         == ["陇", "陇陇", "陇陇*"]
@expect collect(ngram_iterator("陇陇*", order = 3, truncated_start = true)) == ["陇陇*"]

@expect collect(ngram_iterator("", order = 1))                              == []

# -------------------------------------------------------------------------------------------------------------------------
# feature vector tests
# -------------------------------------------------------------------------------------------------------------------------
lines = (Array{String})[]
for l in filelines("data/test.txt")
  tokens = split(strip(l), r"\s+")
  push!(lines, tokens)
end

bkg = make_background(lines)
@expect stats(bkg, "d")       == 19.0
@expect stats(bkg, unk_token) == 1e10

bkg = make_background(lines, mincount = 2)
@expect bkg["d"]              == bkg[unk_token]
@expect stats(bkg, "d")       == 19.0
@expect stats(bkg, unk_token) == 19.0

@info "bkg[c]    = $(stats(bkg, "c"))"
@expect sparse_count(lines[1], bkg)   == sparsevec((Int64=>Float64)[ bkg["a"] => 1.0, bkg["b"] => 1.0, bkg["c"] => 1.0], vocab_size(bkg))
@expect sparse_count(lines[end], bkg) == sparsevec((Int64=>Float64)[ bkg[unk_token] => 1.0 ], vocab_size(bkg))

@info "sparse[c] = $(sparse_count(lines[1], bkg)[2])"
@expect norm(sparse_count(lines[1], bkg), bkg)[2] == 3.166666666666667
@info "normed[c] = $(sparse_count(lines[1], bkg)[2] / stats(bkg, "c"))"

include("lid.jl")
include("topic.jl")


