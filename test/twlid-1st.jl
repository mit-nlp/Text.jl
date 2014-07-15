using Text, Stage, Ollam
using DataStructures

# -------------------------------------------------------------------------------------------------------------------------
# Twitter LID
# -------------------------------------------------------------------------------------------------------------------------
lang_map = ["brazil"   => "pt",
            "china"    => "zh",
            "france"   => "fr",
            "germany"  => "de",
            "iran"     => "fa",
            "japan"    => "ja",
            "malaysia" => "ms",
            "mexico"   => "es",
            "portugal" => "pt",
            "russia"   => "ru",
            "spain"    => "es",
            "taiwan"   => "zh",
            "usa"      => "en"
            ]

xsection = DefaultDict(String, Vector{String}, () -> (String)[])

for (base, lang) in lang_map
  for l in filelines("data/twitter/$base.tsv.gz")
    tweet = split(l, '\t')
    text  = twenglish_cleaner(tweet[7])
    if text != ""
      push!(xsection[lang], text)
    end
  end
end

total = 0
for k in keys(xsection)
  @info @sprintf("%-30s %10d", k, length(xsection[k]))
  total += length(xsection[k])
end
@sep
@info @sprintf("%-30s %10d", "total", total)

train       = vcat([ map(fvt -> fvt, xsection[k][1:min(end-1200, 25000)]) for k in keys(xsection) ]...)
train_truth = vcat([ map(fvt -> k,   xsection[k][1:min(end-1200, 25000)]) for k in keys(xsection) ]...)
test        = vcat([ map(fvt -> fvt, xsection[k][min(end-1200, 25001):end]) for k in keys(xsection) ]...)
test_truth  = vcat([ map(fvt -> k,   xsection[k][min(end-1200, 25001):end]) for k in keys(xsection) ]...)
@info "size of train: $(length(train)), size of test: $(length(test))"

bkgmodel, fextractor, model = tc_train(train, train_truth, lid_iterating_tokenizer, mincount = 2, cutoff = 1e10, 
                                       trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 20, k = 12, C = 0.01, average = true),
                                       iteration_method = :eager)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
res     = test_classification(model, lazy_map(x -> fextractor(lid_iterating_tokenizer(x)), test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0
@info @sprintf("mira test set error rate: %7.3f", res)
print_confusion_matrix(confmat)
#@expect abs(res - 0.596) < 0.01

# List specific errors and filter correct results
#filtered       = String[]
#filtered_truth = String[]
filt = open("filtered.txt", "w")
for (text, t) in zip(test, test_truth)
  fv      = fextractor(lid_iterating_tokenizer(text))
  scores  = score(model, fv)
  bidx, b = best(scores)
  if model.index_class[bidx] == t
    #@debug "ERROR: (ref: $t, hyp: $(model.index_class[bidx])) $text"
  #else
    #push!(filtered, text)
    #push!(filtered_truth, t)
    println(filt, "$t\t$text")
  end
end
close(filt)

# save the model
f = open("model", "w")
serialize(f, bkgmodel)
serialize(f, fextractor)
serialize(f, model)
close(f)
