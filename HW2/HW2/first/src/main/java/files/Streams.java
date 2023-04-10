package files;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class Streams {
	/**
	 * Read from an InputStream until a quote character (") is found, then read
	 * until another quote character is found and return the bytes in between the two quotes. 
	 * If no quote character was found return null, if only one, return the bytes from the quote to the end of the stream.
	 * @param in
	 * @return A list containing the bytes between the first occurrence of a quote character and the second.
	 */
	public static List<Byte> getQuoted(InputStream in) throws IOException {
		try {
			int currentBit = in.read();
			char currentChar;

			while (currentBit != -1){ //run until the end of the stream
				currentChar = (char) currentBit;

				if(currentChar == '\"'){
					List<Byte> outputList = new ArrayList<Byte>();
					currentBit = in.read();

					while(currentBit != -1){
						currentChar = (char) currentBit;

						if(currentChar == '\"') {
							return outputList;
						} else {
							outputList.add((byte)currentBit);
						}
						currentBit = in.read();
					}
					return outputList;
				} else {
					currentBit = in.read();
				}
			}
		} catch (IOException e){
			e.printStackTrace();
		}

		return null;
	}

	/**
	 * Read from the input until a specific string is read, return the string read up to (not including) the endMark.
	 * @param in the Reader to read from
	 * @param endMark the string indicating to stop reading. 
	 * @return The string read up to (not including) the endMark (if the endMark is not found, return up to the end of the stream).
	 */
	public static String readUntil(Reader in, String endMark) throws IOException {
		int endMarkLength = endMark.length();
		char[] arrayToCheck = new char[endMarkLength];
		StringBuilder outputString = new StringBuilder();

		try {
			int readOutput = in.read(arrayToCheck);

			while (readOutput != -1){
				if (endMark.equals(new String(arrayToCheck))){
					return outputString.toString();
				}

				outputString.append(arrayToCheck[0]);
				for (int i = 0; i < endMarkLength - 1; i++){
					arrayToCheck[i] = arrayToCheck[i + 1];
				}

				arrayToCheck[endMarkLength - 1] = 0;
				readOutput = in.read(arrayToCheck, endMarkLength - 1, 1);
			}

		} catch (IOException e){
			e.printStackTrace();
		}

		outputString.append(arrayToCheck);
		return outputString.toString();
	}
	
	/**
	 * Copy bytes from input to output, ignoring all occurrences of badByte.
	 * @param in
	 * @param out
	 * @param badByte
	 */
	public static void filterOut(InputStream in, OutputStream out, byte badByte) throws IOException {
		try {
			int currentByte = in.read();

			while(currentByte != -1){
				if((byte) currentByte != badByte){
					out.write(currentByte);
				}
				currentByte = in.read();
			}
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
	/**
	 * Read a 40-bit (unsigned) integer from the stream and return it. The number is represented as five bytes, 
	 * with the most-significant byte first. 
	 * If the stream ends before 5 bytes are read, return -1.
	 * @param in
	 * @return the number read from the stream
	 */
	public static long readNumber(InputStream in) throws IOException {
		long outputNumber = 0;
		int currentByte;

		try {
			for(int i = 0; i < 5; i++){
				currentByte = in.read();
				if(currentByte == -1){
					return -1;
				}

				outputNumber = outputNumber << 8;
				outputNumber = outputNumber + currentByte;
			}
		} catch (IOException e){
			e.printStackTrace();
		}

		return outputNumber;
	}
}
