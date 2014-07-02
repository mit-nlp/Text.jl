using Text
using Base.Test, Stage, Ollam, DataStructures

const logger = Log(STDERR)

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
# feature extraction
# -------------------------------------------------------------------------------------------------------------------------
@test ngrams(["a", "b", "c"], order = 3) == ["a", "a b", "a b c"]
@test ngrams(["a", "b", "c"], order = 3, truncated_start = true) == ["a b c"]

@test ngrams(["a", "b", "c"], order = 2) == ["a", "a b", "b c"]
@test ngrams(["a", "b", "c"], order = 2, truncated_start = true) == ["a b", "b c"]

@test ngrams(["a", "b", "c"], order = 1) == ["a", "b", "c"]
@test ngrams(["a", "b", "c"], order = 1, truncated_start = true) == ["a", "b", "c"]

@test ngrams(["a"], order = 3) == ["a"]
@test ngrams(["a"], order = 3, truncated_start = true) == []

@test ngrams("abc", order = 3) == ["a", "ab", "abc"]
@test ngrams("abc", order = 3, truncated_start = true) == ["abc"]

@test ngrams("abc", order = 2) == ["a", "ab", "bc"]
@test ngrams("abc", order = 2, truncated_start = true) == ["ab", "bc"]

@test ngrams("abc", order = 1) == ["a", "b", "c"]
@test ngrams("abc", order = 1, truncated_start = true) == ["a", "b", "c"]

@test ngrams("a", order = 3) == ["a"]
@test ngrams("ab", order = 3) == ["a", "ab"]
@test ngrams("abcd", order = 3) == ["a", "ab", "abc", "bcd"]
@test ngrams("a", order = 3, truncated_start = true) == []
@test ngrams("ab", order = 3, truncated_start = true) == []
@test ngrams("abcd", order = 3, truncated_start = true) == ["abc", "bcd"]

# feature vector tests
lines = (Array{String})[]
for l in filelines("data/test.txt")
  tokens = split(strip(l), r"\s+")
  push!(lines, tokens)
end

bkg = make_background(lines)
@test stats(bkg, "d") == 19.0
@test stats(bkg, unk_token) == 1e10

bkg = make_background(lines, mincount = 2)
@test bkg["d"] == bkg[unk_token]
@test stats(bkg, "d") == 19.0
@test stats(bkg, unk_token) == 19.0

@info logger "bkg[c]    = $(stats(bkg, "c"))"
@test sparse_count(lines[1], bkg) == sparsevec((Int64=>Float64)[ bkg["a"] => 1.0, bkg["b"] => 1.0, bkg["c"] => 1.0], vocab_size(bkg))
@test sparse_count(lines[end], bkg) == sparsevec((Int64=>Float64)[ bkg[unk_token] => 1.0 ], vocab_size(bkg))

@info logger "sparse[c] = $(sparse_count(lines[1], bkg)[2])"
@test norm(sparse_count(lines[1], bkg), bkg)[2] == 3.166666666666667
@info logger "normed[c] = $(sparse_count(lines[1], bkg)[2] / stats(bkg, "c"))"

# -------------------------------------------------------------------------------------------------------------------------
# LID
# -------------------------------------------------------------------------------------------------------------------------
train       = map(l -> split(chomp(l), '\t')[2], filelines("data/nus-sms/train.tsv.gz"))
train_truth = map(l -> split(chomp(l), '\t')[1], filelines("data/nus-sms/train.tsv.gz"))
test        = map(l -> split(chomp(l), '\t')[2], filelines("data/nus-sms/test.tsv.gz"))
test_truth  = map(l -> split(chomp(l), '\t')[1], filelines("data/nus-sms/test.tsv.gz"))

@info logger "train: $(length(train)), test: $(length(test))"

bkgmodel, fextractor, model = lid_train(train, train_truth, lid_tokenizer,
                                        trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 2, k = 2, C = 0.01, average = true),
                                        iteration_method = :eager)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
res     = test_classification(model, lazy_map(fextractor, test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0
@info logger @sprintf("mira test set error rate: %7.3f", res)
print_confusion_matrix(confmat)
@test abs(res - 0.700) < 0.03


