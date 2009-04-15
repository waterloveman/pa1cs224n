package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.Pair;

import java.util.*;
import java.lang.*;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class ValidInterpTrigramLanguageModel implements LanguageModel {

  private static final String START = "<S>";
  private static final String STOP = "</S>";
  
  private Counter<String> unigramCounter;
  private CounterMap<String, String> bigramCounter;
  private CounterMap<Pair<String,String>, String> trigramCounter;
  private HashMap<Integer, Integer> freqOfUnigram;
  private HashMap<Integer, Integer> freqOfBigram;
  private HashMap<Integer, Integer> freqOfTrigram;
  private double unigramTotal, bigramTotal, trigramTotal;
  private double alpha1, alpha2, alpha3;
  private double unigramNorm, bigramNorm, trigramNorm;


  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public ValidInterpTrigramLanguageModel() {
    unigramCounter = new Counter<String>();
    bigramCounter = new CounterMap<String, String>();
    trigramCounter = new CounterMap<Pair<String, String>, String>();
    freqOfUnigram = new HashMap<Integer, Integer>();
    freqOfBigram = new HashMap<Integer, Integer>();
    freqOfTrigram = new HashMap<Integer, Integer>();
    unigramNorm = 1.0;
    bigramNorm = 1.0;
    trigramNorm = 1.0;
    unigramTotal = Double.NaN;
    bigramTotal = Double.NaN;
    trigramTotal = Double.NaN;
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public ValidInterpTrigramLanguageModel(Collection<List<String>> sentences) {
    this();
    train(sentences);
  }


  // -----------------------------------------------------------------------

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public void train(Collection<List<String>> sentences) {
    unigramCounter = new Counter<String>();
    bigramCounter = new CounterMap<String, String>();
    trigramCounter = new CounterMap<Pair<String, String>, String>();

    for (List<String> sentence : sentences) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(0, START);
      stoppedSentence.add(0, START);
      stoppedSentence.add(STOP);
      Pair<String, String> prevWords = new Pair<String, String>(START, START);
      int count = 0;
      for (String word : stoppedSentence) {
        unigramCounter.incrementCount(word, 1.0);
	if (count == 0) {
	  count++;
	  continue;
	}
	bigramCounter.incrementCount(prevWords.getSecond(), word, 1.0);
	if (count == 1) {
	  count++;
	  continue;
	}
	trigramCounter.incrementCount(prevWords, word, 1.0);
	prevWords = new Pair<String, String>(prevWords.getSecond(), word);
      }
    }
    unigramTotal = unigramCounter.totalCount();
    bigramTotal = bigramCounter.totalCount();
    trigramTotal = trigramCounter.totalCount();

    freqOfTrigram.put(0, (int) trigramTotal);
    Iterator<Pair<String,String>> prevWords = trigramCounter.keySet().iterator();
    while (prevWords.hasNext()) {
      Pair<String,String> curPrevWord = prevWords.next();
      Iterator<String> words = trigramCounter.getCounter(curPrevWord).keySet().iterator();
      while (words.hasNext()) {
	String curWord = words.next();
	int count = (int) trigramCounter.getCount(curPrevWord, curWord);
	if (freqOfTrigram.containsKey(count))
	  freqOfTrigram.put(count, freqOfTrigram.get(count) + 1);
	else
	  freqOfTrigram.put(count, 1);
      }
    }

    //normalize total
    Iterator<Integer> trigramCounts = freqOfTrigram.keySet().iterator();
    trigramNorm = 0;
    while(trigramCounts.hasNext())
    {
    	int i = trigramCounts.next();
    	if (i == 0)
	  trigramNorm += 1;
    	else
    	  trigramNorm += freqOfTrigram.get(i)* goodTuring(freqOfTrigram, i);
    }
    trigramNorm = trigramTotal / trigramNorm;
  }

  private double goodTuring(HashMap<Integer,Integer> freqOfFreq, double c) {
      double k = 10;
      if(c > k) {
	      return c;
      }
      double N1 = freqOfFreq.get(1);
      double Nc = freqOfFreq.get((int) c);
      if(c == 0) {
	      return N1 / Nc;
      }
      double Nc1 = freqOfFreq.get((int) (c+1));
      double Nk1 = freqOfFreq.get((int) (k+1));
      double cStar = ((c+1)*(Nc1/Nc) - c*(k+1)*Nk1/N1) / (1 - (k+1)*Nk1/N1);
      return cStar;
  }

  public void validate(Collection<List<String>> validationData) {

    Counter<String> backupUnigramCounter = copyCounter(unigramCounter);
    CounterMap<String,String> backupBigramCounter = copyCounterMap(bigramCounter);
    CounterMap<Pair<String,String>,String> backupTrigramCounter = copyCounterMap(trigramCounter);
    train(validationData);

    double delta = 0.05;
    double maxLL = Double.NEGATIVE_INFINITY;
    double maxAlpha1 = 1.0;
    double maxAlpha2 = 0.0;
    double maxAlpha3 = 0.0;
    for (double inc1 = 0.0; inc1 <= 1.0; inc1 += delta) {
      alpha1 = inc1;
      for (double inc2 = 0.0; inc2 <= 1.0 - inc1; inc2 += delta) {
	alpha2 = inc2;
	alpha3 = 1.0 - inc1 - inc2;
	double logLike = calculateLogLike(validationData);
	if (logLike > maxLL) {
	  maxAlpha1 = alpha1;
	  maxAlpha2 = alpha2;
	  maxAlpha3 = alpha3;
	  maxLL = logLike;
	}
      }
    }
    alpha1 = maxAlpha1;
    alpha2 = maxAlpha2;
    alpha3 = maxAlpha3;
    System.out.println("Alpha values: " + alpha1 + " " + alpha2 + " " + alpha3);

   // for (int i = 0; i < 50; i++) {
   //   System.out.println(alpha1 + " " + alpha2 + " " + alpha3);
   //   double c1 = 0.0;
   //   double c2 = 0.0;
   //   double c3 = 0.0;
   //   Iterator<Pair<String,String>> iter = trigramCounter.keySet().iterator();
   //   while(iter.hasNext()) {
   //     Pair<String,String> prevWords = iter.next();
   //     Counter<String> wordCounter = trigramCounter.getCounter(prevWords);
   //     Iterator<String> wordIter = wordCounter.keySet().iterator();
   //     while(wordIter.hasNext()) {
   //       String curWord = wordIter.next();
   //       double trigramCount = trigramCounter.getCount(prevWords, curWord);
   //       double probTrigramMLE = getTrigramProbability(prevWords, curWord);
   //       double probBigramMLE = getBigramProbability(prevWords.getSecond(), curWord);
   //       double probUnigramMLE = getUnigramProbability(curWord);
   //       double c1Numerator = trigramCount * alpha1 * probTrigramMLE;
   //       double c2Numerator = trigramCount * alpha2 * probBigramMLE;
   //       double c3Numerator = trigramCount * alpha3 * probUnigramMLE;
   //       double denominator = (alpha1 * probTrigramMLE) + (alpha2 * probBigramMLE) + (alpha3 * probUnigramMLE);
   //       c1 += (c1Numerator / denominator);
   //       c2 += (c2Numerator / denominator);
   //       c3 += (c3Numerator / denominator);
   //     }
   //   }
   //   alpha1 = c1 / (c1 + c2 + c3);
   //   alpha2 = c2 / (c1 + c2 + c3);
   //   alpha3 = c3 / (c1 + c2 + c3);
   // }
    unigramCounter = backupUnigramCounter;
    bigramCounter = backupBigramCounter;
    trigramCounter = backupTrigramCounter;
  }

  private double calculateLogLike(Collection<List<String>> data) {
    double sum = 0.0;
    double numSentences = 0.0;
    for (List<String> sentence : data) {
      for (int i = 2; i < sentence.size(); i++) {

	sum += Math.log(getProbability(new Pair<String,String>(sentence.get(i-2), sentence.get(i-1)), sentence.get(i)));
      }
      numSentences += 1.0;
    }
    return sum / numSentences;
  }



  // -----------------------------------------------------------------------

   private Counter<String> copyCounter(Counter<String> counter) {
    Counter<String> newCounter = new Counter<String>();
    Iterator<String> iter = counter.keySet().iterator();
    while(iter.hasNext()) {
      String curWord = iter.next();
      newCounter.setCount(curWord, counter.getCount(curWord));
    }

    return newCounter;
  }

  private <T> CounterMap<T,String> copyCounterMap(CounterMap<T,String> counterMap) {
    CounterMap<T,String> newCounterMap = new CounterMap<T,String>();
    Iterator<T> iter = counterMap.keySet().iterator();
    while(iter.hasNext()) {
      T curWord = iter.next();
      Iterator<String> wordIter = counterMap.getCounter(curWord).keySet().iterator();
      while (wordIter.hasNext()) {
	String curNextWord = wordIter.next();
	newCounterMap.setCount(curWord, curNextWord, counterMap.getCount(curWord, curNextWord));
      }
    }

    return newCounterMap;
  }

  // -----------------------------------------------------------------------

  private double getUnigramProbability(String word) {
    double count = unigramCounter.getCount(word);
    if (count == 0) 
      return (unigramCounter.size() * 0.75) / unigramTotal;
    else
      return (count - 0.75) / unigramTotal;
  }

  private double getAlpha(String prevWord) {
    double wordTotal = unigramCounter.getCount(prevWord);
    double alphaDiff = 0.0;
    Iterator<String> iter = bigramCounter.getCounter(prevWord).keySet().iterator();
    while (iter.hasNext()) {
      String word = iter.next();
      double numerator = bigramCounter.getCount(prevWord, word) - 0.75;
      alphaDiff += (numerator / wordTotal);
    }
    return 1 - alphaDiff;
  }

  private double getBigramProbability(String prevWord, String word) {
    double unigramCount = unigramCounter.getCount(prevWord);
    double bigramCount = bigramCounter.getCount(prevWord, word);
    if (bigramCount == 0) {
      double unigramSum = 0.0;
      Iterator<String> iter = unigramCounter.keySet().iterator();
      while (iter.hasNext()) {
	String curWord = iter.next();
	if (bigramCounter.getCount(prevWord, curWord) == 0)
	  unigramSum += getUnigramProbability(curWord);
      }
      return getAlpha(prevWord) * getUnigramProbability(word) / unigramSum;
    }
    else
      return (bigramCount - 0.75) / unigramCount;
  }

  private double getTrigramProbability(Pair<String, String> prevWords, String word) {
    double count = trigramCounter.getCount(prevWords, word);
    count = goodTuring(freqOfTrigram, count);
    return count * trigramNorm / trigramTotal;
  }

  private double getProbability(Pair<String, String> prevWords, String word) {
    double trigramProb = getTrigramProbability(prevWords, word);
    double bigramProb = getBigramProbability(prevWords.getSecond(), word);
    double unigramProb = getUnigramProbability(word);
    return (alpha1 * trigramProb) + (alpha2 * bigramProb) + (alpha3 * unigramProb);
  }


  /**
   * Returns the probability, according to the model, of the word specified
   * by the argument sentence and index.  Smoothing is used, so that all
   * words get positive probability, even if they have not been seen
   * before.
   */
  public double getWordProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    Pair<String, String> prevWords = new Pair<String, String>(sentence.get(index - 2), sentence.get(index - 1));
    return getProbability(prevWords, word);
  }

  /**
   * Returns the probability, according to the model, of the specified
   * sentence.  This is the product of the probabilities of each word in
   * the sentence (including a final stop token).
   */
  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedSentence = new ArrayList<String>(sentence);
    stoppedSentence.add(0, START);
    stoppedSentence.add(0, START);
    stoppedSentence.add(STOP);
    double logProb = 0.0;
    for (int index = 2; index < stoppedSentence.size(); index++) {
      logProb += Math.log(getWordProbability(stoppedSentence, index));
    }
    return Math.exp(logProb);
  }

  /**
   * checks if the probability distribution properly sums up to 1
   */
  public double checkModel() {
    Random generator = new Random();
    double highestVarianceSum = 1.0; // Keep track of which sum differs from 1.0 the most

    int numWordsToCheck = 500; // Totally arbitrary number!
    int counter = 0;

    Object[] keySetWords = trigramCounter.keySet().toArray();
    for (int i = 0; i < numWordsToCheck; i++) {
      int randomIndex = generator.nextInt(keySetWords.length);
      Pair<String, String> prevWords = (Pair<String, String>)keySetWords[randomIndex];

      double sum = 0.0;
      Counter<String> curCounter = trigramCounter.getCounter(prevWords);
      for (String word : curCounter.keySet()) {
	sum += getTrigramProbability(prevWords, word);
      }

      //sum += 1.0 / (curCounter.totalCount() + 1.0);

      if (Math.abs(sum - 1.0) > Math.abs(highestVarianceSum - 1.0))
	highestVarianceSum = sum;
    }
    return highestVarianceSum;
  }
  
  /**
   * Returns a random word sampled according to the model.  A simple
   * "roulette-wheel" approach is used: first we generate a sample uniform
   * on [0, 1]; then we step through the vocabulary eating up probability
   * mass until we reach our sample.
   */
  public String generateWord(Pair<String, String> prevWords) {
    double sample = Math.random();
    double sum = 0.0;
    Counter<String> counter = trigramCounter.getCounter(prevWords);
    for (String word : counter.keySet()) {
      sum += counter.getCount(word) / counter.totalCount();
      if (sum > sample) {
        return word;
      }
    }
    return "*UNKNOWN*";   // a little probability mass was reserved for unknowns
  }

  /**
   * Returns a random sentence sampled according to the model.  We generate
   * words until the stop token is generated, and return the concatenation.
   */
  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    Pair<String, String> prevWords = new Pair<String, String>(START, START);
    String word = generateWord(prevWords);
    while (!word.equals(STOP)) {
      sentence.add(word);
      prevWords = new Pair<String, String>(prevWords.getSecond(), word);
      word = generateWord(prevWords);
    }
    return sentence;
  }

}


