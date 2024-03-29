package cs224n.langmodel;

import cs224n.util.Counter;
import cs224n.util.CounterMap;

import java.util.*;
import java.lang.*;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class ValidInterpBigramLanguageModel implements LanguageModel {

  private static final String START = "<S>";
  private static final String STOP = "</S>";
  
  private Counter<String> unigramCounter;
  private CounterMap<String, String> bigramCounter;
  private double unigramTotal, bigramTotal;
  private double alpha1, alpha2;

  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public ValidInterpBigramLanguageModel() {
    unigramCounter = new Counter<String>();
    bigramCounter = new CounterMap<String, String>();
    unigramTotal = Double.NaN;
    bigramTotal = Double.NaN;
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public ValidInterpBigramLanguageModel(Collection<List<String>> sentences) {
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

    for (List<String> sentence : sentences) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(0, START);
      stoppedSentence.add(STOP);
      String prevWord = START;
      for (String word : stoppedSentence) {
        unigramCounter.incrementCount(word, 1.0);
	if (word == START) continue;
	bigramCounter.incrementCount(prevWord, word, 1.0);
	prevWord = word;
      }
    }
    unigramTotal = unigramCounter.totalCount();
    bigramTotal = bigramCounter.totalCount();
  }

  public void validate(Collection<List<String>> validationData) {
    // Initialize weights
    //alpha1 = 0.25;
    //alpha2 = 0.75;

    Counter<String> backupUnigramCounter = copyCounter(unigramCounter);
    CounterMap<String,String> backupBigramCounter = copyCounterMap(bigramCounter);
    train(validationData);

    double delta = 0.05;
    double maxLL = Double.NEGATIVE_INFINITY;
    double maxAlpha1 = 0.0;
    double maxAlpha2 = 1.0;
    for (double inc = 0.0; inc <= 1.0; inc += delta) {
      alpha1 = inc;
      alpha2 = 1.0 - inc;
      double logLike = calculateLogLike(validationData);
      if (logLike > maxLL) {
	maxAlpha1 = alpha1;
	maxAlpha2 = alpha2;
	maxLL = logLike;
      }
    }
    alpha1 = maxAlpha1;
    alpha2 = maxAlpha2;

    System.out.println("A1 " + alpha1 + " A2" + alpha2);

   // for (int i = 0; i < 50; i++) {
   //   System.out.println(alpha1 + " " + alpha2);
   //   double c1 = 0.0;
   //   double c2 = 0.0;
   //   Iterator<String> iter = bigramCounter.keySet().iterator();
   //   while(iter.hasNext()) {
   //     String prevWord = iter.next();
   //     Counter<String> wordCounter = bigramCounter.getCounter(prevWord);
   //     Iterator<String> wordIter = wordCounter.keySet().iterator();
   //     while(wordIter.hasNext()) {
   //       String curWord = wordIter.next();
   //       double bigramCount = bigramCounter.getCount(prevWord, curWord);
   //       double probBigramMLE = getBigramProbability(prevWord, curWord);
   //       double probUnigramMLE = getUnigramProbability(curWord);
   //       double c1Numerator = bigramCount * alpha1 * probBigramMLE;
   //       double c2Numerator = bigramCount * alpha2 * probUnigramMLE;
   //       double denominator = (alpha1 * probBigramMLE) + (alpha2 * probUnigramMLE);
   //       c1 += (c1Numerator / denominator);
   //       c2 += (c2Numerator / denominator);
   //     }
   //   }
   //   alpha1 = c1 / (c1 + c2);
   //   alpha2 = c2 / (c1 + c2);
   // }
    unigramCounter = backupUnigramCounter;
    bigramCounter = backupBigramCounter;
    unigramTotal = unigramCounter.totalCount();
    bigramTotal = bigramCounter.totalCount();
  }
  // -----------------------------------------------------------------------
  
  private double calculateLogLike(Collection<List<String>> data) {
    double sum = 0.0;
    double numSentences = 0.0;
    for (List<String> sentence : data) {
      for (int i = 1; i < sentence.size(); i++) {
	sum += Math.log(getProbability(sentence.get(i-1), sentence.get(i)));
      }
      numSentences += 1.0;
    }
    return sum / numSentences;
  }

  private Counter<String> copyCounter(Counter<String> counter) {
    Counter<String> newCounter = new Counter<String>();
    Iterator<String> iter = counter.keySet().iterator();
    while(iter.hasNext()) {
      String curWord = iter.next();
      newCounter.setCount(curWord, counter.getCount(curWord));
    }

    return newCounter;
  }

  private CounterMap<String,String> copyCounterMap(CounterMap<String,String> counterMap) {
    CounterMap<String,String> newCounterMap = new CounterMap<String,String>();
    Iterator<String> iter = counterMap.keySet().iterator();
    while(iter.hasNext()) {
      String curWord = iter.next();
      Iterator<String> wordIter = counterMap.getCounter(curWord).keySet().iterator();
      while (wordIter.hasNext()) {
	String curNextWord = wordIter.next();
	newCounterMap.setCount(curWord, curNextWord, counterMap.getCount(curWord, curNextWord));
      }
    }

    return newCounterMap;
  }

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

  private double getProbability(String prevWord, String word) {
    double bigramProb = getBigramProbability(prevWord, word);
    double unigramProb = getUnigramProbability(word);
    return (alpha1 * bigramProb) + (alpha2 * unigramProb);
  }

  /**
   * Returns the probability, according to the model, of the word specified
   * by the argument sentence and index.  Smoothing is used, so that all
   * words get positive probability, even if they have not been seen
   * before.
   */
  public double getWordProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    String prevWord = sentence.get(index - 1);
    return getProbability(prevWord, word);
  }

  /**
   * Returns the probability, according to the model, of the specified
   * sentence.  This is the product of the probabilities of each word in
   * the sentence (including a final stop token).
   */
  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedSentence = new ArrayList<String>(sentence);
    stoppedSentence.add(0, START);
    stoppedSentence.add(STOP);
    double logProb = 0.0;
    for (int index = 1; index < stoppedSentence.size(); index++) {
      logProb += Math.log(getWordProbability(stoppedSentence, index));
    }
    return Math.exp(logProb);
  }

  public double checkModel() {
    return (alpha1 * checkBigramModel()) + (alpha2 * checkUnigramModel());
  }

  private double checkBigramModel() {
    Random generator = new Random();
    double highestVarianceSum = 1.0; // Keep track of which sum differs from 1.0 the most

    int numWordsToCheck = 500; // Totally arbitrary number!
    int counter = 0;

    String[] keySetWords = bigramCounter.keySet().toArray(new String[0]);
    for (int i = 0; i < numWordsToCheck; i++) {
      int randomIndex = generator.nextInt(keySetWords.length);
      String prevWord = keySetWords[randomIndex];

      double sum = 0.0;
      Counter<String> curCounter = bigramCounter.getCounter(prevWord);
      for (String word : curCounter.keySet()) {
	sum += getBigramProbability(prevWord, word);
      }

      // Add on discounted mass
      sum +=  getAlpha(prevWord);

      if (Math.abs(sum - 1.0) > Math.abs(highestVarianceSum - 1.0))
	highestVarianceSum = sum;
    }
    return highestVarianceSum;
  }

  private double checkUnigramModel() {
    double sum = 0.0;

    for (String word : unigramCounter.keySet()) {
      sum += getUnigramProbability(word);
    }
    
    sum += (unigramCounter.size() * 0.75) / (unigramTotal);

    return sum;
  }
  
  /**
   * Returns a random word sampled according to the model.  A simple
   * "roulette-wheel" approach is used: first we generate a sample uniform
   * on [0, 1]; then we step through the vocabulary eating up probability
   * mass until we reach our sample.
   */
  public String generateWord(String prevWord) {
    double sample = Math.random();
    double sum = 0.0;
    Counter<String> counter = bigramCounter.getCounter(prevWord);
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
    String word = generateWord(START);
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateWord(word);
    }
    return sentence;
  }

}


