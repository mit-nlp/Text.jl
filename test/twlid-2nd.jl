using Text, Stage, Ollam
using DataStructures

const url_pattern     = r"http://[^\s]*"
const hashtag_pattern = r"^#.*$"
const mention_pattern = r"^@.*$"

function clean_tweet(tw)
  w = replace(strip(tw), url_pattern, "--url--")
  while true
    x = replace_html_entities(w)
    if x != w
      w = x
    else
      break
    end
  end
  if ismatch(r".*&gt;.*", w)
    println("w: $w")
    exit(1)
  end
  return w
end

# -------------------------------------------------------------------------------------------------------------------------
# Twitter LID
# -------------------------------------------------------------------------------------------------------------------------
xsection = DefaultDict(String, Vector{String}, () -> (String)[])

for (class, text) in lazy_map(ln -> split(ln, '\t'), filelines("filtered.txt"))
  if text != "" && text != "--url--"
    push!(xsection[class], strip(text))
  end
end

total = 0
for k in keys(xsection)
  @info @sprintf("%-30s %10d", k, length(xsection[k]))
  total += length(xsection[k])
end
@sep
@info @sprintf("%-30s %10d", "total", total)

train       = vcat([ map(fvt -> fvt, xsection[k][1:min(end-600, 65000)]) for k in keys(xsection) ]...)
train_truth = vcat([ map(fvt -> k,   xsection[k][1:min(end-600, 65000)]) for k in keys(xsection) ]...)
test        = vcat([ map(fvt -> fvt, xsection[k][min(end-600, int(end*0.95)):end]) for k in keys(xsection) ]...)
test_truth  = vcat([ map(fvt -> k,   xsection[k][min(end-600, int(end*0.95)):end]) for k in keys(xsection) ]...)
@info "size of train: $(length(train)), size of test: $(length(test))"

bkgmodel, fextractor, model = tc_train(train, train_truth, lid_iterating_tokenizer, mincount = 2, cutoff = 1e10, 
                                       trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 20, k = 12, C = 0.01, average = true),
                                       iteration_method = :eager)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
res     = test_classification(model, lazy_map(x -> fextractor(lid_iterating_tokenizer(x)), test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0
@info @sprintf("mira test set error rate: %7.3f", res)
print_confusion_matrix(confmat)

# List specific errors
for (text, t) in zip(test, test_truth)
  fv      = fextractor(lid_iterating_tokenizer(text))
  scores  = score(model, fv)
  bidx, b = best(scores)
  if model.index_class[bidx] != t
    @debug "ERROR: (ref: $t, hyp: $(model.index_class[bidx])) $text"
  end
end

# save the model
f = open("flt-model", "w")
serialize(f, bkgmodel)
serialize(f, fextractor)
serialize(f, model)
close(f)

