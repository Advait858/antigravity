/**
 * useCanister Hook - Real ICP Connection
 * NO MOCKS. Connects to local replica or mainnet.
 */

import { useState, useCallback, useEffect } from 'react';
import { Actor, HttpAgent, Identity } from '@dfinity/agent';
import { AuthClient } from '@dfinity/auth-client';
import { Principal } from '@dfinity/principal';

// Configuration
const CANISTER_ID = 'bkyz2-fmaaa-aaaaa-qaaaq-cai'; // Local replica default
const HOST = 'http://127.0.0.1:8080';
const II_URL = 'https://identity.ic0.app';

// Candid Interface (Updated for Iron-Clad)
const idlFactory = ({ IDL }: { IDL: any }) => {
    return IDL.Service({
        get_version: IDL.Func([], [IDL.Text], ['query']),
        get_health: IDL.Func([], [IDL.Text], ['query']),
        get_dashboard_state: IDL.Func([], [IDL.Text], ['query']),
        get_cycle_balance: IDL.Func([], [IDL.Text], ['query']),
        trigger_tick: IDL.Func([], [IDL.Text], []),
    });
};

export interface UseCanisterResult {
    isConnected: boolean;
    principal: string | null;
    isAuthenticated: boolean;
    login: () => Promise<void>;
    logout: () => Promise<void>;
    queryCanister: (method: string, args?: any[]) => Promise<any>;
    updateCanister: (method: string, args?: any[]) => Promise<any>;
}

export function useCanister(): UseCanisterResult {
    const [authClient, setAuthClient] = useState<AuthClient | null>(null);
    const [actor, setActor] = useState<any>(null);
    const [principal, setPrincipal] = useState<string | null>(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isConnected, setIsConnected] = useState(false);

    // Create actor with optional identity
    const createActor = async (identity?: Identity) => {
        try {
            const agent = new HttpAgent({
                host: HOST,
                identity,
            });

            // Fetch root key for local development (NOT for mainnet!)
            if (process.env.NODE_ENV !== 'production') {
                await agent.fetchRootKey();
            }

            const newActor = Actor.createActor(idlFactory, {
                agent,
                canisterId: CANISTER_ID,
            });

            setActor(newActor);
            setIsConnected(true);
        } catch (err) {
            console.error('Failed to create actor:', err);
            setIsConnected(false);
        }
    };

    // Initialize
    useEffect(() => {
        const init = async () => {
            try {
                const client = await AuthClient.create();
                setAuthClient(client);

                const isAuth = await client.isAuthenticated();
                setIsAuthenticated(isAuth);

                if (isAuth) {
                    const identity = client.getIdentity();
                    setPrincipal(identity.getPrincipal().toString());
                    await createActor(identity);
                } else {
                    await createActor(); // Anonymous
                }
            } catch (err) {
                console.error('Failed to initialize auth:', err);
                // Still try to create anonymous actor
                await createActor();
            }
        };

        init();
    }, []);

    // Login with Internet Identity
    const login = useCallback(async () => {
        if (!authClient) return;

        await authClient.login({
            identityProvider: II_URL,
            onSuccess: async () => {
                const identity = authClient.getIdentity();
                setPrincipal(identity.getPrincipal().toString());
                setIsAuthenticated(true);
                await createActor(identity);
            },
            onError: (error) => {
                console.error('Login failed:', error);
            },
        });
    }, [authClient]);

    // Logout
    const logout = useCallback(async () => {
        if (!authClient) return;

        await authClient.logout();
        setPrincipal(null);
        setIsAuthenticated(false);
        await createActor();
    }, [authClient]);

    // Query canister (read-only)
    const queryCanister = useCallback(
        async (method: string, args: any[] = []) => {
            if (!actor) {
                throw new Error('Actor not initialized');
            }

            try {
                const result = await (actor as any)[method](...args);
                return result;
            } catch (err) {
                console.error(`Query ${method} failed:`, err);
                throw err;
            }
        },
        [actor]
    );

    // Update canister (mutations)
    const updateCanister = useCallback(
        async (method: string, args: any[] = []) => {
            if (!actor) {
                throw new Error('Actor not initialized');
            }

            try {
                const result = await (actor as any)[method](...args);
                return result;
            } catch (err) {
                console.error(`Update ${method} failed:`, err);
                throw err;
            }
        },
        [actor]
    );

    return {
        isConnected,
        principal,
        isAuthenticated,
        login,
        logout,
        queryCanister,
        updateCanister,
    };
}
