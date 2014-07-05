using Text
using Base.Test, Stage, Ollam, DataStructures

const logger = Log(STDERR)

all       = lazy_map(l -> split(chomp(l), '\t')[2], filelines("data/saic-sms/all.tsv.gz"))
all_truth = map(l -> split(chomp(l), '\t')[1], filelines("data/saic-sms/all.tsv.gz"))

# -------------------------------------------------------------------------------------------------------------------------
# representative split
# -------------------------------------------------------------------------------------------------------------------------
xsection = DefaultDict(String, Vector{(String, String)}, () -> ((String, String))[])
for fvt in zip(all, all_truth)
  push!(xsection[fvt[2]], fvt)
end

total = 0
for k in keys(xsection)
  @info logger @sprintf("%-30s %10d", k, length(xsection[k]))
  total += length(xsection[k])
end
@sep logger
@info logger @sprintf("%-30s %10d", "total", total)

test        = vcat([ map(fvt -> fvt[1], xsection[k][1:int(length(xsection[k])*0.15)]) for k in keys(xsection) ]...)
test_truth  = vcat([ map(fvt -> fvt[2], xsection[k][1:int(length(xsection[k])*0.15)]) for k in keys(xsection) ]...)
train       = vcat([ map(fvt -> fvt[1], xsection[k][int(length(xsection[k])*0.15)+1:end]) for k in keys(xsection) ]...)
train_truth = vcat([ map(fvt -> fvt[2], xsection[k][int(length(xsection[k])*0.15)+1:end]) for k in keys(xsection) ]...)
@info logger "train: $(length(train)), test: $(length(test))"

# -------------------------------------------------------------------------------------------------------------------------
# LID
# -------------------------------------------------------------------------------------------------------------------------
bkgmodel, fextractor, model = lid_train(train, train_truth, lid_tokenizer,
                                        trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 20, k = 4, C = 0.01, average = true),
                                        iteration_method = :eager)
confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
@info @sprintf("mira test set error rate: %7.3f", test_classification(model, lazy_map(fextractor, test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0)
print_confusion_matrix(confmat)

bkgmodel, fextractor, model = lid_train(collect(train), collect(train_truth), lid_tokenizer,
                                        trainer = (fvs, truth, init_model) -> train_libsvm(fvs, truth, C = 1.0),
                                        iteration_method = :eager)
confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0))
@info @sprintf("svm test set error rate: %7.3f", test_classification(model, lazy_map(fextractor, test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0)
print_confusion_matrix(confmat)


