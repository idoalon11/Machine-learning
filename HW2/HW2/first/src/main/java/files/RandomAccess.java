package files;

import java.io.IOException;
import java.io.RandomAccessFile;

public class RandomAccess {


    public static int temp = 0;

    /**
     * Treat the file as an array of (unsigned) 8-bit values and sort them
     * in-place using a bubble-sort algorithm.
     * You may not read the whole file into memory!
     *
     * @param file
     */
    public static void sortBytes(RandomAccessFile file) throws IOException {
        boolean fileNeedsSorting = true;
        int currentByte;
        int nextByte;

        try {
            while (fileNeedsSorting) {
                file.seek(0); // Placing the pointer on the beginning of the file
                fileNeedsSorting = false;
                currentByte = file.read();

                if (currentByte == -1) {
                    return;
                }

                nextByte = file.read();
                while ((currentByte != -1) && (nextByte != -1)) {
                    if (currentByte > nextByte) { //Swapping between the 2 bytes
                        file.seek(file.getFilePointer() - 2);
                        file.write(nextByte);
                        file.write(currentByte);
                        fileNeedsSorting = true;
                    } else {
                        currentByte = nextByte;
                    }
                    nextByte = file.read();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Treat the file as an array of unsigned 24-bit values (stored MSB first) and sort
     * them in-place using a bubble-sort algorithm.
     * You may not read the whole file into memory!
     *
     * @param file
     * @throws IOException
     */
    public static void sortTriBytes(RandomAccessFile file) throws IOException {
        boolean fileNeedsSorting = true;
        int currentTriBytes;
        int nextTriBytes;

        try {
            while (fileNeedsSorting) {
                file.seek(0);
                fileNeedsSorting = false;
                currentTriBytes = 0;
                nextTriBytes = 0;

                //Reading the current Tri-Number
                currentTriBytes = readingTriNumber(currentTriBytes, file);

                //Reading the next Tri-Number in order to compare it to the current one.
                if (temp != -1) {
                    nextTriBytes = readingTriNumber(nextTriBytes, file);
                }

                while ((temp != -1) && (currentTriBytes != -1) && (nextTriBytes != -1)) {
                    if (currentTriBytes > nextTriBytes) {
                        file.seek(file.getFilePointer() - 6);
                        file.write(nextTriBytes >> 16);
                        file.write((nextTriBytes >> 8) & 255);
                        file.write(nextTriBytes & 255);
                        file.write(currentTriBytes >> 16);
                        file.write((currentTriBytes >> 8) & 255);
                        file.write(currentTriBytes & 255);
                        fileNeedsSorting = true;
                    } else {
                        currentTriBytes = nextTriBytes;
                    }
                    nextTriBytes = 0;
                    nextTriBytes = readingTriNumber(nextTriBytes, file);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int readingTriNumber(int triNumber, RandomAccessFile file) throws IOException {
        for (int i = 2; i >= 0; i--) {
            temp = file.read();
            if (temp == -1) {
                break;
            }
            triNumber += temp << (i * 8);
        }
        return triNumber;
    }
}
