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
export english_tokenizer, twenglish_tokenizer, replace_html_entities, twenglish_cleaner

# -------------------------------------------------------------------------------------------------------------------------
# Tokenizers
# -------------------------------------------------------------------------------------------------------------------------
const punct_word      = r"^[\p{P}\p{Po}\p{Sm}]*(.*?)[\p{P}\p{Po}\p{Sm}]*$"
const english_space   = r"[\s\p{Zs}_-]+"
const url_pattern     = r"http://[^\s]*"
const hashtag_pattern = r"^#.*$"
const mention_pattern = r"^@.*$"

function replace_html_entities(s :: AbstractString)
  replace(s, r"&[^;]+?;", s -> s in keys(html_entity_table) ? html_entity_table[s] : s)
end

function pattern_replace(w :: AbstractString)
  if ismatch(r"^[+-]?\p{Sc}\d+([.,]\d+)*$", w) return "--currency--"
  elseif ismatch(r"^[+-]?\d+([.,]\d+)*%$", w) return "--percent--"
  elseif ismatch(r"^[+-]?\d+([.,]\d+)*$", w) return "--number--"
  elseif ismatch(r"^(https?|ftp):.*$"i, w) return "--url--"
  elseif ismatch(r"\d+:\d+(am|pm)$"i, w) return "--time--"
  elseif ismatch(r"\d+(am|pm)$"i, w) return "--time--"
  elseif ismatch(r"\d+[-/]\d+$", w) return "--date--"
  elseif ismatch(r"\d+[-/]\d+[-/]\d+$", w) return "--date--"
  else return w 
  end
end

function prereplace(sent :: AbstractString)
  r = replace(sent, r"n't\b", " not")
  r = replace(r, r"'s\b", " s's")
  r = replace(r, r"'d\b", " d'd")
end


function english_tokenizer(s :: AbstractString)
  return [ 
    begin 
      m = match(punct_word, w)
      nw = (m == nothing ? w : m.captures[1])
      pattern_replace(nw)
    end for w in filter(x -> !ismatch(r"^[\p{P}\p{C}\p{S})]+$", x), split(prereplace(strip(normalize_string(s, :NFKC))), english_space))
  ]
end

function twenglish_tokenizer(s :: AbstractString)
  return [ 
    begin 
      m = match(r"^(\p{P}*)(.*?)\p{P}*$", w)
      nw = (m == nothing) ? w : (m.captures[1] == "#" ? (m.captures[1] * m.captures[2]) : m.captures[2])
      patternReplace(nw) 
    end for w in filter(x -> !ismatch(r"^(\p{P}|\p{S})+$", x) && !ismatch(r"^@.*$", x), split(strip(s), r"\s+"))
  ]
end

function twenglish_cleaner(tw :: AbstractString; urls = true, hashtags = true, mentions = true)
  ctw = replace(normalize_string(tw, :NFKC), default_space, " ")
  ctw = urls ? replace(ctw, url_pattern, "\u0030\u20E3") : ctw

  while true
    x = replace_html_entities(ctw)
    if x != ctw
      ctw = x
    else
      break
    end
  end

  words = split(ctw)
  words = hashtags ? [ ismatch(hashtag_pattern, w) ? "\u0023\u20E3" : w for w in words ] : words
  words = mentions ? [ ismatch(hashtag_pattern, w) ? "\u0031\u20E3" : w for w in words ] : words

  return join(words, " ")
end
