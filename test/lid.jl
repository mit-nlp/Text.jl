# -------------------------------------------------------------------------------------------------------------------------
# LID
# -------------------------------------------------------------------------------------------------------------------------
train       = map(l -> split(chomp(l), '\t')[2], filelines("data/nus-sms/train.tsv.gz"))
train_truth = map(l -> split(chomp(l), '\t')[1], filelines("data/nus-sms/train.tsv.gz"))
test        = map(l -> split(chomp(l), '\t')[2], filelines("data/nus-sms/test.tsv.gz"))
test_truth  = map(l -> split(chomp(l), '\t')[1], filelines("data/nus-sms/test.tsv.gz"))

@info "train: $(length(train)), test: $(length(test))"
for t in train
  @test lid_tokenizer(t) == collect(lid_iterating_tokenizer(t))
end

bkgmodel, fextractor, model = tc_train(train, train_truth, lid_iterating_tokenizer, mincount = 2, cutoff = 1e10, 
                                       trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 20, k = 2, C = 0.01, average = true),
                                       iteration_method = :eager)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
res     = test_classification(model, lazy_map(x -> fextractor(lid_iterating_tokenizer(x)), test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0
@info @sprintf("mira test set error rate: %7.3f", res)
print_confusion_matrix(confmat)
@expect abs(res - 0.596) < 0.01

# List specific errors
# for (text, t) in zip(test, test_truth)
#   fv      = fextractor(lid_iterating_tokenizer(text))
#   scores  = score(model, fv)
#   bidx, b = best(scores)
#   if model.index_class[bidx] != t
#     @debug "ERROR: (ref: $t, hyp: $(model.index_class[bidx])) $text"
#   end
# end
