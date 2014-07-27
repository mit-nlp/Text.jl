function text(fn) 
  res = ""
  for l in map(l -> chomp(l), eachline(`iconv -f latin1 -t utf8 $fn`))
    ismatch(r"^M[ -`]{60}$", l) && continue
    if res != ""
      res *= " " * l
    else
      res = l
    end
  end
  return res
end

function getinstances(dir)
  docs   = String[]
  truth  = String[]

  for t in filter(d -> d != "." && d != "..", readdir(dir))
    for d in filter(d -> d != "." && d != "..", readdir("$dir/$t"))
      push!(docs, "$dir/$t/$d")
      names = split(t, ".")
      push!(truth, names[end-1][1:1] * "." * names[end][1:min(4, end)])
    end
  end
  docs, truth
end

function tokenize(text)
  cleaned = english_tokenizer(text) # lowercase(text))
  res = filter(w -> !(lowercase(w) in english_stoplist), cleaned)
  res
end
tokenize_file(fn) = tokenize(text(fn))

# -------------------------------------------------------------------------------------------------------------------------
# download and prep data
# -------------------------------------------------------------------------------------------------------------------------
getfile("http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz", "20news-bydate.tar.gz", expected_size = 14_464_277)
if !isdir("20ng")
  run(`tar -xzf 20news-bydate.tar.gz`)
  mkdir("20ng")
  mv("20news-bydate-train", "20ng/train")
  mv("20news-bydate-test", "20ng/test")
end
train, train_truth = getinstances("20ng/train")
test, test_truth   = getinstances("20ng/test")
@info "train: $(length(train)), test: $(length(test))"

# -------------------------------------------------------------------------------------------------------------------------
# topic training
# -------------------------------------------------------------------------------------------------------------------------
bkgmodel, fextractor, model = tc_train(train, train_truth, tokenize_file, mincount = 2, cutoff = 1e10,
                                       trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 20, k = 19, C = 0.01, average = true),
                                       iteration_method = :eager)
confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
res     = test_classification(model, lazy_map(x -> fextractor(tokenize_file(x)), test), test_truth, 
                              record = (t, h) -> confmat[t][h] += 1) * 100.0
@info @sprintf("mira test set error rate: %7.3f", res)
print_confusion_matrix(confmat, width = 6)
@expect abs(res - 14.485) < 0.01

# List specific errors
# for (tr, t) in zip(test, test_truth)
#   trtext  = text(tr)
#   fv      = fextractor(tokenize(trtext))
#   scores  = score(model, fv)
#   bidx, b = best(scores)
#   if model.index_class[bidx] != t
#     @debug "ERROR: (ref: $t, hyp: $(model.index_class[bidx])) $(replace(trtext[1:70], r"\s+", " "))"
#   end
# end
