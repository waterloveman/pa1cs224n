package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.*;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.  (That is, we pretend that there is
 * a single unknown word, and that we saw it just once during training.)
 *
 * @author Dan Klein
 */
public class AbsoluteDiscountUnigramLanguageModel implements LanguageModel {

  private static final String STOP = "</S>";
  
  private Counter<String> wordCounter;
  private double total;
  private double diff;
  
  private HashMap<Integer, Integer> freqOfFreq;
  


  // -----------------------------------------------------------------------

  /**
   * Constructs a new, empty unigram language model.
   */
  public AbsoluteDiscountUnigramLanguageModel() {
	freqOfFreq = new HashMap<Integer, Integer>();
    wordCounter = new Counter<String>();
    total = Double.NaN;
  }

  /**
   * Constructs a unigram language model from a collection of sentences.  A
   * special stop token is appended to each sentence, and then the
   * frequencies of all words (including the stop token) over the whole
   * collection of sentences are compiled.
   */
  public AbsoluteDiscountUnigramLanguageModel(Collection<List<String>> sentences) {
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
    wordCounter = new Counter<String>();
    for (List<String> sentence : sentences) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(STOP);
      for (String word : stoppedSentence) {
        wordCounter.incrementCount(word, 1.0);
      }
    }
    total = wordCounter.totalCount();
    freqOfFreq.put(0, (int) total);
    Iterator<String> words = wordCounter.keySet().iterator();
    while(words.hasNext()) {
    	String w = words.next();
    	double d = (wordCounter.getCount(w));
    	int count = (int) d;
    	if(freqOfFreq.containsKey(count)) {
    		freqOfFreq.put(count, freqOfFreq.get(count)+1);
    	}
    	else {
    		freqOfFreq.put(count, 1);
    	}
    }
    //normalize total
    Iterator<Integer> counts = freqOfFreq.keySet().iterator();
    diff = total;
    while(counts.hasNext())
    {
    	int i = counts.next();
    	if (i == 0) {
    		diff -= 1;
    	}
    	else {
    		diff -= freqOfFreq.get(i) * absoluteDiscounting(i);
    	}
    }
//    System.out.println("\n\n");
//    System.out.println("diff: " + diff);
//    for(double d = 0; d < 7; d++) {
//    	System.out.println(d + ": " + absoluteDiscounting(d));
//    }
//    System.out.println("\n\n");
  }
  
  private double absoluteDiscounting(double c) {
	  double discount = 0.75;
	  if(c > 0) {
		  return c - discount;
	  }
	  
	  return diff;
  }

  // -----------------------------------------------------------------------

  private double getWordProbability(String word) {
    double count = wordCounter.getCount(word);
    count = absoluteDiscounting(count);
    return count / total;
  }

  /**
   * Returns the probability, according to the model, of the word specified
   * by the argument sentence and index.  Smoothing is used, so that all
   * words get positive probability, even if they have not been seen
   * before.
   */
  public double getWordProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    return getWordProbability(word);
  }

  /**
   * Returns the probability, according to the model, of the specified
   * sentence.  This is the product of the probabilities of each word in
   * the sentence (including a final stop token).
   */
  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedSentence = new ArrayList<String>(sentence);
    stoppedSentence.add(STOP);
    double probability = 1.0;
    for (int index = 0; index < stoppedSentence.size(); index++) {
      probability *= getWordProbability(stoppedSentence, index);
    }
    return probability;
  }

  /**
   * checks if the probability distribution properly sums up to 1
   */
  public double checkModel() {
    double sum = 0.0;
    // since this is a unigram model, 
    // the event space is everything in the vocabulary (including STOP)
    // and a UNK token

    // this loop goes through the vocabulary (which includes STOP)
    for (String word : wordCounter.keySet()) {
      sum += getWordProbability(word);
    }
    
    // remember to add the UNK. In this AbsoluteDiscountingUnigramLanguageModel
    // we assume there is only one UNK, so we add...
    sum += (diff + 1.0) / (total);
    
    return sum;
  }
  
  /**
   * Returns a random word sampled according to the model.  A simple
   * "roulette-wheel" approach is used: first we generate a sample uniform
   * on [0, 1]; then we step through the vocabulary eating up probability
   * mass until we reach our sample.
   */
  public String generateWord() {
    double sample = Math.random();
    double sum = 0.0;
    for (String word : wordCounter.keySet()) {
      sum += wordCounter.getCount(word) / total;
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
    String word = generateWord();
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateWord();
    }
    return sentence;
  }

public void validate(Collection<List<String>> validationData) {
	// TODO Auto-generated method stub
	
}

}


