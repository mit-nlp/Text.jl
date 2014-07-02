# models.jl
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
export make_background, stats, vocab_size, apply

type BKG
  dict  :: Associative{String, Int32}
  index :: Array{String}
  stats :: Vector{Float64}
end
vocab_size(bkg::BKG) = length(bkg.index)
getindex(bkg::BKG, token :: String) = bkg.dict[token]
stats(bkg::BKG, s::String) = bkg.stats[bkg[s]]

function tfnorm(stats; cutoff = 1e10, squash :: Function = log)
  for i = 1:length(stats)
    stats[i] = min(cutoff, squash(1.0 / stats[i]))
  end

  return stats
end

function apply(bkg::BKG, counts)
  for i in indices(counts)
    counts[i] *= bkg.stats[i]
  end
  return counts
end

function make_background(features; mincount = 1, prune = 0.0, unk = true, norm = stats -> min(1.0 ./ stats, 1e10), logger = Log(STDERR))
  dict = DefaultDict(String, Int32, 0)

  @timer logger "building background dictionary" begin
  # Count
  for fv in features
    for f in fv
      dict[f] += 1
    end
  end

  # clean
  unkcount = 0
  total    = 0
  for (k, v) in dict
    if v < mincount
      unkcount += v
      delete!(dict, k)
    end
    total += v
  end
  dict[unk_token] = unkcount
  end # timer

  # index
  index          = (String)[unk_token]
  rev            = DefaultDict(String, Int32, 1)
  rev[unk_token] = 1
  i              = 2
  @timer logger "building index" begin
  for (k, v) in dict
    if k != unk_token
      push!(index, k)
      rev[k] = i
      i += 1
    end
  end
  end # timer

  # make model vector
  stats = Array(Float64, i-1)
  @timer logger "making bkg vector and pruning" begin
  for (k, v) in dict
    stats[rev[k]] = v
  end
  stats /= total

  # prune
  for i = 1:length(stats)
    if stats[i] < prune
      stats[i] = 0.0
    end
  end
  end # timer

  return BKG(rev, index, norm(stats))
end

