#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <cstdlib>
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

using namespace std;

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
      vector<vector<unsigned>> sentences;
      vector<vector<unsigned>> sentencesPos;
      vector<map<int, int>> sentencesHead;
      vector<map<int, string>> sentencesDeprel;
      unsigned nsentences;

      vector<vector<unsigned>> sentencesDev;
      vector<vector<string>> sentencesStrDev;
      vector<vector<unsigned>> sentencesPosDev;
      vector<map<int, int>> sentencesHeadDev;
      vector<map<int, string>> sentencesDeprelDev;
      unsigned nsentencesDev;

      unsigned nwords;
      unsigned npos;
      unsigned ndeprel;
      unsigned nchars;
      unsigned nactions;

      map<string, unsigned> wordsToInt;
      map<unsigned, string> intToWords;

      map<string, unsigned> posToInt;
      map<unsigned, string> intToPos;

      map<string, unsigned> deprelToInt;
      map<unsigned, string> intToDeprel;

      bool USE_SPELLING = false;
      map<string, unsigned> charsToInt;
      map<unsigned, string> intToChars;

      vector<string> actions;
      map<string, int> ractions;

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

        _add(wordsToInt, intToWords, Corpus::BAD0, nwords);
        _add(wordsToInt, intToWords, Corpus::UNK, nwords);

        _add(charsToInt, intToChars, "UNK", nchars);
        _add(charsToInt, intToChars, "BAD0", nchars);
      }

      static inline unsigned UTF8Len(unsigned char x)
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

      void load_train(const string &file, const string &actionsFile){
        _load(file, sentences, nullptr, sentencesPos, sentencesHead, sentencesDeprel);
        nsentences = sentences.size();
        cerr << "done" << "\n";

        _loadActions(actionsFile, actions, ractions);
        // for (auto a : actions)
        // {
        //   cerr << a << endl;
        // }
        nactions = actions.size();
        cerr << "nactions:" << nactions << endl;
        cerr << "nwords:" << nwords << endl;
        for (unsigned i = 0; i < npos; i++)
        {
          cerr << i << ":" << intToPos[i] << endl;
        } 
      }

      void load_dev(const string &file){
        _load(file, sentencesDev, &sentencesStrDev, sentencesPosDev, sentencesHeadDev, sentencesDeprelDev);
        nsentencesDev = sentencesDev.size();
      }

      inline unsigned get_or_add_word(const string& word){
        _add(wordsToInt, intToWords, word, nwords);
        return wordsToInt[word];
      }

    private:
      bool _add(map<string, unsigned> &toInt, map<unsigned, string> &fromInt, const string &s, unsigned &n){
        if(toInt.find(s)==toInt.end()){
          toInt[s] = n;
          fromInt[n] = s;
          n++;
          return true;
        }
        return false;
      }

      void _load(string file,
                  vector<vector<unsigned>> &sentences,
                  vector<vector<string>> *sentencesStr,
                  vector<vector<unsigned>> &sentencesPos,
                  vector<map<int,int>> &sentencesHead,
                  vector<map<int,string>> &sentencesDeprel
                ){

        sentences.clear();
        if(sentencesStr){sentencesStr->clear();}
        sentencesPos.clear();
        sentencesHead.clear();
        sentencesDeprel.clear();

        ifstream inF(file);
        string line;
        vector<unsigned> current_sent;
        vector<string> current_sent_str;
        vector<unsigned> current_sent_pos;
        map<int,int> current_sent_head;
        map<int,string> current_sent_deprel;
        while(getline(inF, line)){
          vector<string> tokens;
          boost::split(tokens, line, boost::is_any_of("\t"));

          if(line.length()==0){
            if(current_sent.size()>0){
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
            }
            continue;
          }

          if(tokens.size()!=10){
            continue;
          }

          string idx = tokens[0];
          string surface = tokens[1];
//          string lemma = tokens[2];
          string pos = tokens[3];
//          string xpos = tokens[4];
//          string features = tokens[5];
          string head = tokens[6];
          string deprel = tokens[7];


          if (_add(wordsToInt, intToWords, surface, nwords)){
            //if new word, add to char map as well
            unsigned j = 0;
            while (j < surface.length()){
              string wj = "";
              for (unsigned h = j; h < j + UTF8Len(surface[j]); h++){
                wj += surface[h];
              }
              _add(charsToInt, intToChars, wj, nchars);
              j += UTF8Len(surface[j]);
            }
          }
          _add(posToInt, intToPos, pos, npos);
          _add(deprelToInt, intToDeprel, deprel, ndeprel);

          current_sent.push_back(wordsToInt[surface]);
          current_sent_str.push_back(surface);
          current_sent_pos.push_back(posToInt[pos]);
          current_sent_head[atoi(idx.c_str())-1]=atoi(head.c_str())-1;
          current_sent_deprel[atoi(idx.c_str())-1]=deprelToInt[deprel];
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

      void _loadActions(const string &file, vector<string> &actions, map<string, int> &ractions){
        actions.clear();
        ractions.clear();
        ifstream inF(file);
        string line;
        while(getline(inF, line)){
          actions.push_back(line);
          ractions[line]=(actions.size()-1);
        }
        inF.close();
      }
  };
}

#endif
