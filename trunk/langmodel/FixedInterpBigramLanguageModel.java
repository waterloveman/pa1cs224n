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
public class FixedInterpBigramLanguageModel implements LanguageModel {

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
  public FixedInterpBigramLanguageModel() {
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
  public FixedInterpBigramLanguageModel(Collection<List<String>> sentences) {
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
    // Use fixed values for interpolation weighting
    alpha1 = 0.75;
    alpha2 = 0.25;
  }
  // -----------------------------------------------------------------------

  private double getUnigramProbability(String word) {
    double count = unigramCounter.getCount(word);
    if (count == 0) count = 1.0;
    return count / (unigramTotal + 1.0);
  }

  private double getBigramProbability(String prevWord, String word) {
    double unigramCount = unigramCounter.getCount(prevWord);
    double bigramCount = bigramCounter.getCount(prevWord, word);
    if (bigramCount == 0) bigramCount = 1.0;
    return bigramCount / (unigramCount + 1.0) ;
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
    double bigramProb = getBigramProbability(prevWord, word);
    double unigramProb = getUnigramProbability(word);
    return (alpha1 * bigramProb) + (alpha2 * unigramProb);
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

  /**
   * checks if the probability distribution properly sums up to 1
   */
  public double checkModel() {
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

      sum += 1.0 / (bigramCounter.getCounter(prevWord).totalCount() + 1.0);

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


