#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

#include <boost/algorithm/string.hpp>

namespace cpyp
{
  // class Token
  // {
  // public:
  //   unsigned w;
  //   unsigned pos;
  //   int head;
  //   unsigned deprel;

  //   public Token(unsigned _w, unsigned _pos, int _head, unsigned _deprel)
  //   : w(_w), pos(_pos), head(_head), deprel(_deprel)
  //   {}
  // };

  class Corpus
  {
    public:
      std::vector<std::vector<unsigned>> sentences;
      std::vector<std::vector<unsigned>> sentencesPos;
      std::vector<std::map<int, int>> sentencesHead;
      std::vector<std::map<int, std::string>> sentencesDeprel;
      unsigned nsentences;

      std::vector<std::vector<unsigned>> sentencesDev;
      std::vector<std::vector<std::string>> sentencesStrDev;
      std::vector<std::vector<unsigned>> sentencesPosDev;
      std::vector<std::map<int, int>> sentencesHeadDev;
      std::vector<std::map<int, std::string>> sentencesDeprelDev;
      unsigned nsentencesDev;

      unsigned nwords;
      unsigned npos;
      unsigned ndeprel;
      unsigned nchars;
      unsigned nactions;

      std::map<std::string, unsigned> wordsToInt;
      std::map<unsigned, std::string> intToWords;

      std::map<std::string, unsigned> posToInt;
      std::map<unsigned, std::string> intToPos;

      std::map<std::string, unsigned> deprelToInt;
      std::map<unsigned, std::string> intToDeprel;

      bool USE_SPELLING = false;
      std::map<std::string, unsigned> charsToInt;
      std::map<unsigned, std::string> intToChars;

      std::vector<std::string> actions;
      std::map<std::string, int> ractions;

      // String literals
      static constexpr const char* UNK = "UNK";
      static constexpr const char* BAD0 = "<BAD0>";

      Corpus()
      {
        nwords=0;
        npos=0;
        ndeprel=0;
        nactions=0;
        nchars=1;

        _add(wordToInt, intToWords, Corpus::BAD0, nwords);
        _add(wordToInt, intToWords, Corpus::UNK, nwords);

        _add(charsToInt, intToChars, "UNK", nchars);
        _add(charsToInt, intToChars, "BAD0", nchars);
      }

      inline unsigned UTF8Len(unsigned char x)
      {
        if (x < 0x80)
          return 1;
        else if ((x >> 5) == 0x06)
          return 2;
        else if ((x >> 4) == 0x0e)
          return 3;
        else if ((x >> 3) == 0x1e)
          return 4;
        else if ((x >> 2) == 0x3e)
          return 5;
        else if ((x >> 1) == 0x7e)
          return 6;
        else
          return 0;
      }

      void load_train(const std::string &file, const std::string &actionsFile){
        _load(file, sentences, nullptr, sentencesPos, sentencesHead, sentencesDeprel);
        nsentences = sentences.size();
        std::cerr << "done" << "\n";

        _loadActions(actionsFile, actions, ractions);
        for (auto a : actions)
        {
          std::cerr << a << std::endl;
        }
        nactions = actions.size();
        std::cerr << "nactions:" << nactions << std::endl;
        std::cerr << "nwords:" << nwords << std::endl;
        for (unsigned i = 0; i < npos; i++)
        {
          std::cerr << i << ":" << intToPos[i] << std::endl;
        } 
      }

      void load_dev(const std::string &file){
        _load(file, sentencesDev, sentencesStrDev, sentencesPosDev, sentencesHeadDev, sentencesDeprelDev);
        nsentencesDev = sentencesDev.size();
      }

      inline unsigned get_or_add_word(const std::string& word){
        _add(wordsToInt, intToWords, word, nwords);
        return wordsToInt[word];
      }

    private:
      static bool _add(std::map<std::string, unsigned> &toInt, std::map<unsigned, std::string> &fromInt, const std::string &s, int &n){
        if(toInt.find(s)==toInt.end()){
          toInt[s] = n;
          fromInt[n] = s;
          n++;
          return true;
        }
        return false;
      }

      static void _load(std::string file,
                  std::map<int, std::vector<unsigned>> &sentences,
                  std::map<int, std::vector<unsigned>> *sentencesStr,
                  std::map<int, std::vector<unsigned>> &sentencesPos,
                  std::map<int, std::map<int,int>> &sentencesHead,
                  std::map<int, std::map<int,std::string>> &sentencesDeprel
                ){

        sentences.clear();
        if(sentencesStr){sentenceStr->clear();}
        sentencesPos.clear();
        sentencesHead.clear();
        sentencesDeprel.clear();

        std::ifstream inF(file);
        std::string line;
        std::vector<unsigned> current_sent;
        std::vector<unsigned> current_sent_str;
        std::vector<unsigned> current_sent_pos;
        std::map<int,int> current_sent_head;
        std::map<int,std::string> current_sent_deprel;
        while(std::getline(inF, line)){
          std::vector<std::string> tokens;
          boost::split(tokens, line, boost::is_any_of("\t"));

          if(tokens.size()==0 && current_sent.size()>0){
              sentences.push_back(current_sent);
              if(sentencesStr){sentencesStr->push_back(current_sent_str);}
              sentencesPos.push_back(current_sent_pos);
              sentencesHead.push_back(current_sent_head);
              sentencesDeprel.push_back(current_sent_deprel);

              current_sent.clear();
              current_sent_str.clear();
              current_sent_pos.clear();
              current_sent_head.clear();
              current_sent_deprel.clear();
              continue;
          }

          if(tokens.size()!=10){
            continue;
          }

          std::string idx = tokens[0];
          std::string surface = tokens[1];
//          std::string lemma = tokens[2];
          std::string pos = tokens[3];
//          std::string xpos = tokens[4];
//          std::string features = tokens[5];
          std::string head = tokens[6];
          std::string deprel = tokens[7];


          if (_add(wordToInt, intToWords, word, nwords)){
            //if new word, add to char map as well
            unsigned j = 0;
            while (j < word.length()){
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++){
                wj += word[h];
              }
              _add(charsToInt, intToChars, wj, nchars);
              j += UTF8Len(word[j]);
            }
          }
          _add(posToInt, intToPos, pos, npos);
          _add(deprelToInt, intToDeprel, deprel, ndeprel);

          current_sent.push_back(wordToInt[surface]);
          current_sent_str.push_back(surface);
          current_sent_pos.push_back(posToInt[pos]);
          current_sent_head[std::atoi(idx)]=std::atoi(head);
          current_sent_deprel.push_back(delrelToInt[deprel]);
        }

        if(current_sent.size()>0){
          sentences.push_back(current_sent);
          if(sentencesStr){sentencesStr->push_back(current_sent_str);}
          sentencesPos.push_back(current_sent_pos);
          sentencesHead.push_back(current_sent_head);
          sentencesDeprel.push_back(current_sent_deprel);
        }

        inF.close();
      }

      static void _loadActions(const std::string &file, std::vector<std::string> &actions, std::map<std::string, int> &ractions){
        actions.clear();
        ractions.clear();
        std::ifstream inF(file);
        std::line;
        while(std::getline(inF, line)){
          actions.push_back(line);
          ractions[line]=(actions.size()-1);
        }
        inF.close();
      }
  };
}

#endif
