package dict;

import java.io.*;
import java.util.Map;
import java.util.TreeMap;


/**
 * Implements a persistent dictionary that can be held entirely in memory.
 * When flushed, it writes the entire dictionary back to a file.
 *
 * The file format has one keyword per line:
 * <pre>word:def1:def2:def3,...</pre>
 *
 * Note that an empty definition list is allowed (in which case the entry would have the form: <pre>word:</pre>
 *
 * @author talm
 *
 */
public class InMemoryDictionary extends TreeMap<String,String> implements PersistentDictionary  {
	private static final long serialVersionUID = 1L; // (because we're extending a serializable class)
	private final File file;

	public InMemoryDictionary(File dictFile) {
		file = dictFile;
	}

	@Override
	public void open() throws IOException {
		FileReader fileReader;
		BufferedReader bufferedReader = null;
		int separatorIndex;
		String line;
		String key;
		String value;

		try{
			fileReader = new FileReader(this.file.getAbsolutePath());
			bufferedReader = new BufferedReader(fileReader);
			line = bufferedReader.readLine();

			while (line != null){
				separatorIndex = line.indexOf(':');
				key = line.substring(0, separatorIndex);
				value = line.substring(separatorIndex + 1);
				this.put(key, value);
				line = bufferedReader.readLine();
			}
		} catch (java.io.FileNotFoundException e){

		} finally {
			if(bufferedReader != null){
				bufferedReader.close();
			}
		}
	}

	@Override
	public void close() throws IOException {
		FileWriter fileWriter;
		BufferedWriter bufferedWriter = null;
		try {
			fileWriter = new FileWriter(file);
			bufferedWriter = new BufferedWriter(fileWriter);
			for (Map.Entry<String, String> entry : this.entrySet()){
				bufferedWriter.write(entry.getKey() + ":" + entry.getValue() + '\n');
			}
		} finally {
			if(bufferedWriter != null){
				bufferedWriter.close();
			}
		}

	}

}
