import Text "mo:base/Text";
import Array "mo:base/Array";
import Float "mo:base/Float";
import Int "mo:base/Int";
import Nat "mo:base/Nat";
import Time "mo:base/Time";
import Buffer "mo:base/Buffer";
import Iter "mo:base/Iter";
import Debug "mo:base/Debug";
import Cycles "mo:base/ExperimentalCycles";

persistent actor TradingAgent {
    
    // ============================================================
    // TYPES
    // ============================================================
    
    type Trade = {
        id: Nat;
        timestamp: Int;
        action: Text;
        asset: Text;
        amount: Float;
        price: Float;
        reasoning: Text;
    };
    
    // ============================================================
    // STABLE STATE
    // ============================================================
    
    stable var tradeCounter : Nat = 0;
    stable var tradeHistory : [Trade] = [];
    stable var usdBalance : Float = 100000.0;
    stable var positions : [(Text, Float, Float)] = [];
    stable var agentLogs : [Text] = [];
    
    // ============================================================
    // HELPERS
    // ============================================================
    
    private func log(msg: Text) : () {
        let entry = Int.toText(Time.now()) # ": " # msg;
        agentLogs := Array.append(agentLogs, [entry]);
        Debug.print(entry);
    };
    
    private func getPosition(asset: Text) : ?(Float, Float) {
        for ((a, amt, price) in positions.vals()) {
            if (a == asset) { return ?(amt, price); };
        };
        null
    };
    
    private func updatePosition(asset: Text, newAmount: Float, newAvgPrice: Float) : () {
        let buffer = Buffer.Buffer<(Text, Float, Float)>(positions.size());
        var found = false;
        
        for ((a, amt, price) in positions.vals()) {
            if (a == asset) {
                if (newAmount > 0) { buffer.add((a, newAmount, newAvgPrice)); };
                found := true;
            } else {
                buffer.add((a, amt, price));
            };
        };
        
        if (not found and newAmount > 0) {
            buffer.add((asset, newAmount, newAvgPrice));
        };
        
        positions := Buffer.toArray(buffer);
    };
    
    // ============================================================
    // QUERIES
    // ============================================================
    
    public query func get_version() : async Text {
        "Antigravity v2.0 - Hybrid AI Trading Agent"
    };
    
    public query func get_portfolio() : async Text {
        var result = "{\"usd\":" # Float.toText(usdBalance) # ",\"positions\":[";
        var first = true;
        
        for ((asset, amount, avgPrice) in positions.vals()) {
            if (not first) { result #= ","; };
            result #= "{\"asset\":\"" # asset # "\",\"amount\":" # Float.toText(amount) # ",\"avg_price\":" # Float.toText(avgPrice) # "}";
            first := false;
        };
        
        result # "]}"
    };
    
    public query func get_trade_history() : async Text {
        var result = "[";
        var first = true;
        
        for (trade in tradeHistory.vals()) {
            if (not first) { result #= ","; };
            result #= "{\"id\":" # Nat.toText(trade.id) # 
                      ",\"ts\":" # Int.toText(trade.timestamp) #
                      ",\"action\":\"" # trade.action # "\"" #
                      ",\"asset\":\"" # trade.asset # "\"" #
                      ",\"amount\":" # Float.toText(trade.amount) #
                      ",\"price\":" # Float.toText(trade.price) #
                      ",\"reason\":\"" # trade.reasoning # "\"}";
            first := false;
        };
        
        result # "]"
    };
    
    public query func get_logs() : async Text {
        var result = "[";
        var first = true;
        let start = if (agentLogs.size() > 50) { agentLogs.size() - 50 } else { 0 };
        
        for (i in Iter.range(start, agentLogs.size() - 1)) {
            if (not first) { result #= ","; };
            result #= "\"" # agentLogs[i] # "\"";
            first := false;
        };
        
        result # "]"
    };
    
    public query func get_cycles() : async Nat {
        Cycles.balance()
    };
    
    // ============================================================
    // UPDATES (Trade Execution)
    // ============================================================
    
    public func execute_trade(action: Text, asset: Text, amount: Float, price: Float, reasoning: Text) : async Text {
        log("TRADE: " # action # " " # Float.toText(amount) # " " # asset # " @ $" # Float.toText(price));
        
        let tradeValue = amount * price;
        
        if (action == "BUY") {
            if (tradeValue > usdBalance) {
                return "{\"ok\":false,\"err\":\"Insufficient USD\"}";
            };
            
            usdBalance -= tradeValue;
            
            switch (getPosition(asset)) {
                case (?(amt, avgP)) {
                    let total = amt + amount;
                    let newAvg = ((amt * avgP) + (amount * price)) / total;
                    updatePosition(asset, total, newAvg);
                };
                case null {
                    updatePosition(asset, amount, price);
                };
            };
            
        } else if (action == "SELL") {
            switch (getPosition(asset)) {
                case (?(amt, avgP)) {
                    if (amount > amt) {
                        return "{\"ok\":false,\"err\":\"Insufficient \" # asset # \"\"}";
                    };
                    usdBalance += tradeValue;
                    updatePosition(asset, amt - amount, avgP);
                };
                case null {
                    return "{\"ok\":false,\"err\":\"No position\"}";
                };
            };
        } else {
            return "{\"ok\":false,\"err\":\"Invalid action\"}";
        };
        
        let trade : Trade = {
            id = tradeCounter;
            timestamp = Time.now();
            action = action;
            asset = asset;
            amount = amount;
            price = price;
            reasoning = reasoning;
        };
        
        tradeHistory := Array.append(tradeHistory, [trade]);
        tradeCounter += 1;
        
        log("SUCCESS. Balance: $" # Float.toText(usdBalance));
        
        "{\"ok\":true,\"id\":" # Nat.toText(trade.id) # ",\"bal\":" # Float.toText(usdBalance) # "}"
    };
    
    public func reset() : async Text {
        usdBalance := 100000.0;
        positions := [];
        tradeHistory := [];
        tradeCounter := 0;
        agentLogs := [];
        log("RESET to $100k");
        "{\"ok\":true}"
    };
    
    // DEX Hook (future)
    public func swap_dex(from: Text, to: Text, amt: Float) : async Text {
        log("DEX SWAP (sim): " # Float.toText(amt) # " " # from # " -> " # to);
        "{\"ok\":true,\"simulated\":true}"
    };
}
