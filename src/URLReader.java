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
        Trade trade;

        while ( (inputLine = bis.readLine()) != null ) {
            // dont print header
            if (!firstLine) {
                trade = new Trade(inputLine);
                System.out.print(trade.getCurrency() + " ");
                System.out.print(trade.getPrice() + " ");
                System.out.print(trade.getSector() + "\n");
            }
            firstLine = false;
        }
    }
}
