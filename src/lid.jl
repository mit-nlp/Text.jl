# lid.jl
#
# author: Wade Shen
# swade@ll.mit.edu
# Copyright &copy; 2014 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export lid_train, lid_tokenizer

# -------------------------------------------------------------------------------------------------------------------------
# LID
# -------------------------------------------------------------------------------------------------------------------------
function lid_tokenizer(text)
  x   = strip(text)
  res = split(x, default_space)
  ngrams!(res, x, order = 1)
  ngrams!(res, x, order = 2)
  ngrams!(res, x, order = 3)
  return res
end

function lid_features(text, bkgmodel)
  counts = sparse_count(text, bkgmodel)
  counts /= sum(counts)
  return apply(bkgmodel, counts)
end

function lid_train(text, truth, preprocess::Function; cutoff = 1e10, mincount = 2, prune = 0.0, 
                   iteration_method = :lazy,
                   trainer = (fvs, truth, init_model) -> train_mira(fvs, truth, init_model, iterations = 3, average = true),
                   logger = Log(STDERR))
  mapper = iteration_method == :eager ? map : lazy_map

  # define class index
  classes = Dict{String, Int32}()
  i       = 1
  @timer logger "indexing truth" for t in truth
    if !(t in keys(classes))
      classes[t] = i
      i += 1
    end
  end

  # prep model
  @timer logger "preprocessing input"     preprocessed_text = mapper(preprocess, text)
  @timer logger "making background model" bkgmodel          = make_background(preprocessed_text, mincount = mincount, prune = prune, 
                                                                              norm = stats -> tfnorm(stats, squash = sqrt, cutoff = 1e10))
  @timer logger "making feature vectors"  fvs               = mapper(text -> lid_features(text, bkgmodel), preprocessed_text)
  @timer logger "initializating model"    init_model        = LinearModel(classes, vocab_size(bkgmodel))
  model             = trainer(fvs, truth, init_model)
  
  return bkgmodel, text -> lid_features(preprocess(text), bkgmodel), model
end
