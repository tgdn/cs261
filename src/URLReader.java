import java.net.*;
import java.io.*;

public class URLReader {

    public static void main(String[] args) throws Exception {
        Socket socketStream = new Socket("cs261.dcs.warwick.ac.uk", 80);

        BufferedReader bis = new BufferedReader(
            new InputStreamReader(socketStream.getInputStream())
        );
        String inputLine;
        boolean firstLine = true;

        while ( (inputLine = bis.readLine()) != null ) {
            // dont print header
            if (!firstLine) {
                System.out.println(inputLine);
            }
            firstLine = false;
        }
    }
}
