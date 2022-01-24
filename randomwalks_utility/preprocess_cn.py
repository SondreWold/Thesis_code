import codecs

default_dict = {
  "antonyms": "is an antonym of",
  "isA": "is a",
  "mannerOf": "is a manner of",
  "synonyms": "is a synonym of"
  }

LAMA_relations = [
  "atLocation",
  "capableOf",
  "causes",
  "causesDesire",
  "desires",
  "hasA",
  "hasPrerequisite",
  "hasProperty",
  "hasSubevent",
  "isA",
  "locatedNear",
  "madeOf",
  "motivatedByGoal",
  "partOf",
  "receivesAction",
  "usedFor"
] 

LAMA_dict = {
  "atLocation": "is at",
  "capableOf": "is capable of",
  "causes": "causes",
  "causesDesire": "causes desire of",
  "desires": "desires",
  "hasA": "has a",
  "hasPrerequisite": "has prerequisite",
  "hasProperty": "has property",
  "hasSubevent": "has subevent",
  "isA": "is a",
  "locatedNear": "is located near",
  "madeOf": "is made of",
  "motivatedByGoal": "is motivated by",
  "partOf": "is part of",
  "receivesAction": "recieves",
  "usedFor": "is used for"
} 

def create_joined_assertions_for_random_walks(paths=[], relation_dict = default_dict, output_path="../data/concept_net/randomwalks/cn_assertions_filtered.tsv"):

  all_assertions = []
  for path in paths:
    relation = path.split("cn_")[1].split(".txt")[0]
    nl_relation = relation_dict[relation]
    with codecs.open(path, "r", "utf8") as f:
      for line in f.readlines():
        word_a, word_b = line.strip().split("\t")
        full_assertion = [word_a, word_b, nl_relation]
        all_assertions.append(full_assertion)
        # Handle bidirectionality
        if relation == "antonyms" or relation == "synonyms":
          full_assertion_b = [word_b, word_a, nl_relation]
          all_assertions.append(full_assertion_b)
  print("In total, we have %d assertions" % len(all_assertions))
  with codecs.open(output_path, "w", "utf8") as out:
    for assertion in all_assertions:
      out.write(assertion[0] + "\t" + assertion[1] + "\t" + assertion[2] + "\n")



def main():
  paths = [f"../data/concept_net/relations/cn_{x}.txt" for x in LAMA_relations]
  relation_dict = LAMA_dict
  create_joined_assertions_for_random_walks(paths=paths, relation_dict=relation_dict)

if __name__ == "__main__":
  main()
