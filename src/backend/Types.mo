/**
 * Shared Types for Antigravity Canisters
 * HTTP outcall types for management canister
 */

module {
    public type HttpHeader = { name: Text; value: Text };
    public type HttpMethod = { #get; #post; #head };
    
    public type HttpRequestArgs = {
        url: Text;
        max_response_bytes: ?Nat64;
        method: HttpMethod;
        headers: [HttpHeader];
        body: ?Blob;
        transform: ?TransformContext;
    };
    
    public type HttpResponsePayload = {
        status: Nat;
        headers: [HttpHeader];
        body: Blob;
    };
    
    public type TransformArgs = {
        response: HttpResponsePayload;
        context: Blob;
    };
    
    public type TransformContext = {
        function: shared query TransformArgs -> async HttpResponsePayload;
        context: Blob;
    };
    
    // Oracle types
    public type VerificationArtifact = {
        request_url: Text;
        request_timestamp: Int;
        response_status: Nat;
        payload_hash: Text;
        payload_size: Nat;
    };
    
    public type OracleResponse = {
        artifact: VerificationArtifact;
        raw_json: Text;
    };
    
    public type OracleResult = {
        #ok: OracleResponse;
        #err: Text;
    };
}
