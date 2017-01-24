public class Trade {

    protected String time;
    protected String buyer;
    protected String seller;

    protected Float price;
    protected Integer size;

    protected String currency;
    protected String symbol;
    protected String sector;

    protected Float bid; // max price buyer is willing to pay for a stock at given time
    protected Float ask; // min price buyer is willing to sell a stock at given time


    public Trade(String row) {
        String[] splitRow = row.split(",");

        this.time = splitRow[0];
        this.buyer = splitRow[1];
        this.seller = splitRow[2];
        this.price = Float.parseFloat(splitRow[3]);
        this.size = Integer.parseInt(splitRow[4]);
        this.currency = splitRow[5];
        this.symbol = splitRow[6];
        this.sector = splitRow[7];
        this.bid = Float.parseFloat(splitRow[8]);
        this.ask = Float.parseFloat(splitRow[9]);
    }

    public String getTime() {
        return time;
    }

    public Float getPrice() {
        return price;
    }

    public Integer getSize() {
        return size;
    }

    public String getCurrency() {
        return currency;
    }

    public String getSymbol() {
        return symbol;
    }

    public String getSector() {
        return sector;
    }

    public Float getBid() {
        return bid;
    }

    public Float getAsk() {
        return ask;
    }
}
