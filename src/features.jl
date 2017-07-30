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
export ngrams, count, tfnorm, sparse_count, norm, znorm, ngram_iterator, ngrams!

struct NgramStringIterator 
  string :: AbstractString
  order :: Int32
  truncated_start :: Bool
end
Base.iteratorsize(::Type{NgramStringIterator}) = Base.SizeUnknown()

mutable struct StringPosition
  start  :: Int32
  fin    :: Int32
  nth    :: Int32
end

function start(ngi :: NgramStringIterator) 
  if ngi.truncated_start 
    idx = 1
    for i = 1:(ngi.order-1)
      idx = nextind(ngi.string, idx)
    end
    return StringPosition(1, idx, ngi.order)
  else
    return StringPosition(1, 1, 1)
  end
end

done(ngi :: NgramStringIterator, position) = position.fin > endof(ngi.string)
function next(ngi :: NgramStringIterator, position)
  str = make_string(ngi.string, position.start, position.fin)
  if position.nth >= ngi.order
    position.start = nextind(ngi.string, position.start)
  end
  position.nth += 1
  position.fin  = nextind(ngi.string, position.fin)
  return str, position
end

# -------------------------------------------------------------------------------------------------------------------------
# feature extractors
# -------------------------------------------------------------------------------------------------------------------------
make_string(words :: AbstractString, b, e) = SubString(words, b, e)
make_string(words :: Array, b, e) = join(words[b:e], " ")

function ngrams(words::Array; order = 2, truncated_start = false)
  ret = AbstractString[]

  if !truncated_start
    for wi = 1:min(order - 1, length(words))
      push!(ret, make_string(words, 1, wi))
    end
  end

  for wi = order:length(words)
    push!(ret, make_string(words, wi - order + 1, wi))
  end
  return ret
end

function ngrams(words :: AbstractString; order = 2, truncated_start = false)
  ret = AbstractString[]
  return ngrams!(ret, words, order = order, truncated_start = truncated_start)
end

function ngrams!(ret :: Array, words :: AbstractString; order = 2, truncated_start = false)
  for x in ngram_iterator(words, order = order, truncated_start = truncated_start)
    push!(ret, x)
  end
  return ret
end

ngram_iterator(words :: AbstractString; order = 2, truncated_start = false) = NgramStringIterator(words, order, truncated_start)

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
  map = DefaultDict{AbstractString,Int32}()
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
