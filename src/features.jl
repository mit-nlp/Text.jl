# features.jl
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
export ngrams, count, tfnorm, sparse_count, norm, znorm

# -------------------------------------------------------------------------------------------------------------------------
# feature extractors
# -------------------------------------------------------------------------------------------------------------------------
make_string(words :: String, b, e) = SubString(words, b, e)
make_string(words :: Array, b, e) = join(words[b:e], " ")

function ngrams(words::Array; order = 2, truncated_start = false)
  ret = String[]

  if !truncated_start
    for wi = 1:min(order - 1, length(words))
      push!(ret, make_string(words, 1, wi))
    end
  end

  for wi = order:length(words)
    push!(ret, make_string(words, wi - order + 1, wi)) #join(words[wi-order+1:wi], " "))
  end
  return ret
end

function ngrams(words::String; order = 2, truncated_start = false)
  ret = String[]
  wi  = 1

  for i = 1:(order - 1)
    if !truncated_start
      push!(ret, make_string(words, 1, wi))
    end
    wi = nextind(words, wi)
    if wi > endof(words)
      break
    end
  end

  pwi = 1
  while wi <= endof(words)
    push!(ret, make_string(words, pwi, wi))
    pwi = nextind(words, pwi)
    wi  = nextind(words, wi)
  end
  return ret
end

function ngrams!(ret :: Array, words :: String; order = 2, truncated_start = false)
  wi  = 1

  for i = 1:(order - 1)
    if !truncated_start
      push!(ret, make_string(words, 1, wi))
    end
    wi = nextind(words, wi)
    if wi > endof(words)
      break
    end
  end

  pwi = 1
  while wi <= endof(words)
    push!(ret, make_string(words, pwi, wi))
    pwi = nextind(words, pwi)
    wi  = nextind(words, wi)
  end
  return ret
end

# -------------------------------------------------------------------------------------------------------------------------
# feature vector operations
# -------------------------------------------------------------------------------------------------------------------------
function sparse_count(text, bkg)
  vec = spzeros(vocab_size(bkg), 1)
  for token in text
    vec[bkg[token]] += 1
  end
  return vec
end


function dict_count(tokens)
  map = DefaultDict{String,Int32}()
  for w in tokens
    map[w] += 1
  end
end

function znorm(vector, mean, std)
  @devec ret = (vector - mean) ./ std
  return ret
end

function norm(counts, bkg; cutoff = 0.0)
  for r in counts.rowval
    counts[r] *= bkg.stats[r]
  end

  # prune
  nn = nonzeros(counts)
  for i = 1:length(nn)
    if nn[i] < cutoff
      nn[i] = 0.0
    end
  end

  return counts
end
