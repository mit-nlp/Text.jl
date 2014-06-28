# tokenizers.jl
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
export english_tokenizer, twenglish_tokenizer

# -------------------------------------------------------------------------------------------------------------------------
# Tokenizers
# -------------------------------------------------------------------------------------------------------------------------

function patternReplace(w :: String)
  if ismatch(r"^\d+$", w) return "--number--"
  elseif ismatch(r"^http:.*$", w) return "--url--"
  elseif ismatch(r"\d+:\d+(am|pm)$", w) return "--time--"
  elseif ismatch(r"\d+(am|pm)$", w) return "--time--"
  elseif ismatch(r"\d+[-/]\d+$", w) return "--date--"
  elseif ismatch(r"\d+[-/]\d+[-/]\d+$", w) return "--date--"
  else return w 
  end
end

function english_tokenizer(s :: String)
  return [ 
    begin 
      m = match(r"^\p{P}*(.*?)\p{P}*$", w)
      nw = (m == nothing ? w : m.captures[1])
      patternReplace(nw) # ismatch(r"^\d+$", nw) ? "--number--" : nw
    end for w in filter(x -> !ismatch(r"^(\p{P}|\p{S})+$", x), split(strip(s), r"\s+"))
  ]
end

function twenglish_tokenizer(s :: String)
  return [ 
    begin 
      m = match(r"^(\p{P}*)(.*?)\p{P}*$", w)
      nw = (m == nothing) ? w : (m.captures[1] == "#" ? (m.captures[1] * m.captures[2]) : m.captures[2])
      patternReplace(nw) 
    end for w in filter(x -> !ismatch(r"^(\p{P}|\p{S})+$", x) && !ismatch(r"^@.*$", x), split(strip(s), r"\s+"))
  ]
end

