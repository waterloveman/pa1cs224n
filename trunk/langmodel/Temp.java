import java.io.*;

public class Temp {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		BufferedReader in = new BufferedReader(new FileReader("data/europarl.lowercased.ascii.short.notused"));
		BufferedWriter out = new BufferedWriter(new FileWriter("data/europarl.100k.txt"));
		String line = "";
		int i = 0;
		while((line = in.readLine()) != null) {
			if(i > 100000) {
				break;
			}
			i++;
			out.write(line + "\n");
		}
		out.close();
		in.close();
	}

}
