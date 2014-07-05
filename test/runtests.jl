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
@test l == 6717

l = 0
for i in lazy_map(x -> split(strip(x), '\t')[2], filelines("data/nus-sms/test.tsv.gz"))
  l += 1
  if (l == 6717)
    println("6717: ", i)
  end
end
@test l == 6717
end

# -------------------------------------------------------------------------------------------------------------------------
# tokenizers
# -------------------------------------------------------------------------------------------------------------------------
@test english_tokenizer("  test  ")     == ["test"]
@test english_tokenizer("  test.")      == ["test"]
@test english_tokenizer("\"test.\"")    == ["test"]
@test english_tokenizer(">test")        == ["test"]
@test english_tokenizer(">test<")       == ["test"]
@test english_tokenizer("\$100.0")      == ["--currency--"]
@test english_tokenizer("\$100")        == ["--currency--"]
@test english_tokenizer("100.0")        == ["--number--"]
@test english_tokenizer("https://test") == ["--url--"]
@test english_tokenizer("ftp://test")   == ["--url--"]
@test english_tokenizer("http://test")  == ["--url--"]
@test english_tokenizer("ice-cream")    == ["ice", "cream"]
@test english_tokenizer("ice  cream")   == ["ice", "cream"]
@test english_tokenizer("ice--cream")   == ["ice", "cream"]
@test english_tokenizer("ice_cream")    == ["ice", "cream"]
@test english_tokenizer("don't")        == ["do", "not"]
@test english_tokenizer("don't be")     == ["do", "not", "be"]
@test english_tokenizer("cat's")        == ["cat", "s's"]
@test english_tokenizer("how'd")        == ["how", "d'd"]

# -------------------------------------------------------------------------------------------------------------------------
# feature extraction
# -------------------------------------------------------------------------------------------------------------------------
# ngrams from arrays
@test ngrams(["a", "b", "c"], order = 3)                         == ["a", "a b", "a b c"]
@test ngrams(["a", "b", "c"], order = 3, truncated_start = true) == ["a b c"]

@test ngrams(["a", "b", "c"], order = 2)                         == ["a", "a b", "b c"]
@test ngrams(["a", "b", "c"], order = 2, truncated_start = true) == ["a b", "b c"]

@test ngrams(["a", "b", "c"], order = 1)                         == ["a", "b", "c"]
@test ngrams(["a", "b", "c"], order = 1, truncated_start = true) == ["a", "b", "c"]

@test ngrams(["a"], order = 3)                                   == ["a"]
@test ngrams(["a"], order = 3, truncated_start = true)           == []

# ngrams from strings
@test ngrams("abc", order = 3)                           == ["a", "ab", "abc"]
@test ngrams("abc", order = 3, truncated_start = true)   == ["abc"]

@test ngrams("abc", order = 2)                           == ["a", "ab", "bc"]
@test ngrams("abc", order = 2, truncated_start = true)   == ["ab", "bc"]

@test ngrams("abc", order = 1)                           == ["a", "b", "c"]
@test ngrams("abc", order = 1, truncated_start = true)   == ["a", "b", "c"]

@test ngrams("a", order = 3)                             == ["a"]
@test ngrams("ab", order = 3)                            == ["a", "ab"]
@test ngrams("abcd", order = 3)                          == ["a", "ab", "abc", "bcd"]
@test ngrams("a", order = 3, truncated_start = true)     == []
@test ngrams("ab", order = 3, truncated_start = true)    == []
@test ngrams("abcd", order = 3, truncated_start = true)  == ["abc", "bcd"]

@test ngrams("是的", order = 1)                          == ["是", "的"]
@test ngrams("是的", order = 2)                          == ["是", "是的"]
@test ngrams("是的", order = 3)                          == ["是", "是的"]
@test ngrams("是的", order = 3, truncated_start = true)  == []

@test ngrams("陇陇*", order = 1)                         == ["陇", "陇", "*"]
@test ngrams("陇陇*", order = 2)                         == ["陇", "陇陇", "陇*"]
@test ngrams("陇陇*", order = 3)                         == ["陇", "陇陇", "陇陇*"]
@test ngrams("陇陇*", order = 3, truncated_start = true) == ["陇陇*"]

@test ngrams("", order = 1)                              == []

# ngram iterator
@test collect(ngram_iterator("abc", order = 3))                           == ["a", "ab", "abc"]
@test collect(ngram_iterator("abc", order = 3, truncated_start = true))   == ["abc"]

@test collect(ngram_iterator("abc", order = 2))                           == ["a", "ab", "bc"]
@test collect(ngram_iterator("abc", order = 2, truncated_start = true))   == ["ab", "bc"]

@test collect(ngram_iterator("abc", order = 1))                           == ["a", "b", "c"]
@test collect(ngram_iterator("abc", order = 1, truncated_start = true))   == ["a", "b", "c"]

@test collect(ngram_iterator("a", order = 3))                             == ["a"]
@test collect(ngram_iterator("ab", order = 3))                            == ["a", "ab"]
@test collect(ngram_iterator("abcd", order = 3))                          == ["a", "ab", "abc", "bcd"]
@test collect(ngram_iterator("a", order = 3, truncated_start = true))     == []
@test collect(ngram_iterator("ab", order = 3, truncated_start = true))    == []
@test collect(ngram_iterator("abcd", order = 3, truncated_start = true))  == ["abc", "bcd"]

@test collect(ngram_iterator("是的", order = 1))                          == ["是", "的"]
@test collect(ngram_iterator("是的", order = 2))                          == ["是", "是的"]
@test collect(ngram_iterator("是的", order = 3))                          == ["是", "是的"]
@test collect(ngram_iterator("是的", order = 3, truncated_start = true))  == []

@test collect(ngram_iterator("陇陇*", order = 1))                         == ["陇", "陇", "*"]
@test collect(ngram_iterator("陇陇*", order = 2))                         == ["陇", "陇陇", "陇*"]
@test collect(ngram_iterator("陇陇*", order = 3))                         == ["陇", "陇陇", "陇陇*"]
@test collect(ngram_iterator("陇陇*", order = 3, truncated_start = true)) == ["陇陇*"]

@test collect(ngram_iterator("", order = 1))                              == []

# -------------------------------------------------------------------------------------------------------------------------
# feature vector tests
# -------------------------------------------------------------------------------------------------------------------------
lines = (Array{String})[]
for l in filelines("data/test.txt")
  tokens = split(strip(l), r"\s+")
  push!(lines, tokens)
end

bkg = make_background(lines)
@test stats(bkg, "d")       == 19.0
@test stats(bkg, unk_token) == 1e10

bkg = make_background(lines, mincount = 2)
@test bkg["d"]              == bkg[unk_token]
@test stats(bkg, "d")       == 19.0
@test stats(bkg, unk_token) == 19.0

@info "bkg[c]    = $(stats(bkg, "c"))"
@test sparse_count(lines[1], bkg)   == sparsevec((Int64=>Float64)[ bkg["a"] => 1.0, bkg["b"] => 1.0, bkg["c"] => 1.0], vocab_size(bkg))
@test sparse_count(lines[end], bkg) == sparsevec((Int64=>Float64)[ bkg[unk_token] => 1.0 ], vocab_size(bkg))

@info "sparse[c] = $(sparse_count(lines[1], bkg)[2])"
@test norm(sparse_count(lines[1], bkg), bkg)[2] == 3.166666666666667
@info "normed[c] = $(sparse_count(lines[1], bkg)[2] / stats(bkg, "c"))"

include("lid.jl")
include("topic.jl")


