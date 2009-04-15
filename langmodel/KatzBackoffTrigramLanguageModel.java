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
public class KatzBackoffTrigramLanguageModel implements LanguageModel {

  private static final String START = "<S>";
  private static final String STOP = "</S>";
  
  private Counter<String> unigramCounter;
  private CounterMap<String, String> bigramCounter;
  private CounterMap<Pair<String,String>, String> trigramCounter;
  private HashMap<Integer, Integer> freqOfBigram;
  private double unigramTotal, bigramTotal, trigramTotal;
  private double bigramNorm;


  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public KatzBackoffTrigramLanguageModel() {
    unigramCounter = new Counter<String>();
    bigramCounter = new CounterMap<String, String>();
    trigramCounter = new CounterMap<Pair<String, String>, String>();
    freqOfBigram = new HashMap<Integer, Integer>();
    bigramNorm = 1.0;
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
  public KatzBackoffTrigramLanguageModel(Collection<List<String>> sentences) {
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

    freqOfBigram.put(0, (int) bigramTotal);
    Iterator<String> prevWords = bigramCounter.keySet().iterator();
    while (prevWords.hasNext()) {
	String curPrevWord = prevWords.next();
	Iterator<String> words = bigramCounter.getCounter(curPrevWord).keySet().iterator();
	while (words.hasNext()) {
	  String curWord = words.next();
	  int count = (int) bigramCounter.getCount(curPrevWord, curWord);
	  if (freqOfBigram.containsKey(count))
	    freqOfBigram.put(count, freqOfBigram.get(count) + 1);
	  else
	    freqOfBigram.put(count, 1);
	}
    }

    //normalize total
    Iterator<Integer> bigramCounts = freqOfBigram.keySet().iterator();
    bigramNorm = 0;
    while(bigramCounts.hasNext())
    {
    	int i = bigramCounts.next();
    	if (i == 0)
	  bigramNorm += 1;
    	else
    	  bigramNorm += freqOfBigram.get(i)* goodTuring(freqOfBigram, i);
    }
    bigramNorm = bigramTotal / bigramNorm;
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
    // Empty
  }

  // -----------------------------------------------------------------------

  private double getBigramProbability(String prevWord, String word) {
    double count = bigramCounter.getCount(prevWord, word);
    count = goodTuring(freqOfBigram, count);
    return count * bigramNorm / bigramTotal;
  }

  private double getTwoAlpha(Pair<String,String> prevWords) {
    double wordTotal = bigramCounter.getCount(prevWords.getFirst(), prevWords.getSecond());
    double alphaDiff = 0.0;
    Iterator<String> iter = trigramCounter.getCounter(prevWords).keySet().iterator();
    while (iter.hasNext()) {
      String word = iter.next();
      double numerator = trigramCounter.getCount(prevWords, word) - 0.75;
      alphaDiff += (numerator / wordTotal);
    }
    return 1 - alphaDiff;
  }

  private double getTrigramProbability(Pair<String,String> prevWords, String word) {
    double bigramCount = bigramCounter.getCount(prevWords.getFirst(), prevWords.getSecond());
    double trigramCount = trigramCounter.getCount(prevWords, word);
    if (trigramCount == 0) {
      double bigramSum = 0.0;
      Iterator<String> iter = unigramCounter.keySet().iterator();
      while (iter.hasNext()) {
	String curWord = iter.next();
	if (trigramCounter.getCount(prevWords, curWord) == 0)
	  bigramSum += getBigramProbability(prevWords.getSecond(), curWord);
      }
      return getTwoAlpha(prevWords) * getBigramProbability(prevWords.getSecond(), word);
    }
    else
      return (trigramCount - 0.75) / bigramCount;
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
    return getTrigramProbability(prevWords, word);
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

      // Add on discounted mass
      sum += getTwoAlpha(prevWords);

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


