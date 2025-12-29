/**
 * Antigravity Oracle Canister - Production v2.6
 * 
 * HTTPS outcalls to CoinGecko for crypto prices.
 * Compatible with dfx 0.30.1 using persistent actor.
 */

import Text "mo:base/Text";
import Blob "mo:base/Blob";
import Nat "mo:base/Nat";
import Nat64 "mo:base/Nat64";
import Nat32 "mo:base/Nat32";
import Time "mo:base/Time";
import Int "mo:base/Int";
import Iter "mo:base/Iter";
import Error "mo:base/Error";
import Cycles "mo:base/ExperimentalCycles";
import Char "mo:base/Char";
import Types "Types";

persistent actor Oracle {
    
    // Management canister for HTTPS outcalls
    let IC = actor "aaaaa-aa" : actor {
        http_request : Types.HttpRequestArgs -> async Types.HttpResponsePayload;
    };
    
    // State
    stable var totalFetches : Nat = 0;
    stable var lastFetchTime : Int = 0;
    stable var lastRawJson : Text = "";
    
    // Hash function for verification
    private func computeHash(text: Text) : Text {
        var hash : Nat = 5381;
        for (c in text.chars()) {
            hash := ((hash * 33) + Nat32.toNat(Char.toNat32(c))) % 4294967295;
        };
        Nat.toText(hash)
    };
    
    // Transform for deterministic responses
    public query func transform(args: Types.TransformArgs) : async Types.HttpResponsePayload {
        { status = args.response.status; headers = []; body = args.response.body }
    };
    
    // Main fetch function
    public func fetch_prices(assetIds: [Text]) : async Types.OracleResult {
        if (assetIds.size() == 0) { return #err("No assets") };
        
        let ids = Text.join(",", Iter.fromArray(assetIds));
        let url = "https://api.coingecko.com/api/v3/simple/price?ids=" # ids # "&vs_currencies=usd&include_24hr_change=true";
        let reqTime = Time.now();
        
        Cycles.add<system>(50_000_000_000);
        
        try {
            let resp = await IC.http_request({
                url = url;
                max_response_bytes = ?Nat64.fromNat(10000);
                method = #get;
                headers = [{ name = "Accept"; value = "application/json" }];
                body = null;
                transform = ?{ function = transform; context = Blob.fromArray([]) };
            });
            
            if (resp.status != 200) { return #err("HTTP " # Nat.toText(resp.status)) };
            
            let body = switch (Text.decodeUtf8(resp.body)) {
                case (?t) t;
                case null { return #err("UTF-8 error") };
            };
            
            totalFetches += 1;
            lastFetchTime := reqTime;
            lastRawJson := body;
            
            #ok({
                artifact = {
                    request_url = url;
                    request_timestamp = reqTime;
                    response_status = resp.status;
                    payload_hash = computeHash(body);
                    payload_size = Text.size(body);
                };
                raw_json = body;
            })
        } catch (e) {
            #err("Failed: " # Error.message(e))
        }
    };
    
    public func fetch_top_prices() : async Types.OracleResult {
        await fetch_prices(["bitcoin", "ethereum", "internet-computer"])
    };
    
    public func fetch_trading_prices() : async Types.OracleResult {
        await fetch_prices(["bitcoin", "ethereum", "internet-computer", "solana", "ripple", "cardano", "dogecoin", "polkadot", "avalanche-2", "chainlink"])
    };
    
    public query func get_last_json() : async Text { lastRawJson };
    
    public query func get_health() : async Text {
        "{\"status\":\"healthy\",\"version\":\"2.6\",\"fetches\":" # Nat.toText(totalFetches) # ",\"cycles\":" # Nat.toText(Cycles.balance()) # "}"
    };
    
    public query func get_cycles() : async Nat { Cycles.balance() };
}
