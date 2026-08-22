#!/usr/bin/env perl
#
# Least-connections TCP balancer for ninfer-multi.
#
# Why not round-robin: inference requests have wildly different lifetimes. A
# 200k-context generation can occupy a replica for minutes while a short
# completion returns in a second, so rotating blindly buries one card while
# another idles. Least-connections tracks live sessions per replica and always
# hands the next request to the least busy engine.
#
# Why TCP and not HTTP-aware: ninfer-serve speaks OpenAI + Anthropic HTTP with
# SSE streaming. Relaying bytes keeps streaming, chunked encoding, keep-alive
# semantics, and both API surfaces working without parsing or buffering them.
# The tradeoff is that a keep-alive connection stays pinned to one replica for
# its lifetime, which is correct for streaming clients.
#
# Dead-replica ejection: there is no health probe — the balancer sees the truth
# directly, because a dead replica refuses connections. After
# EJECT_THRESHOLD consecutive refused sessions a port is excluded from
# selection for EJECT_COOLDOWN seconds (30, matching the outer LiteLLM
# router's cooldown); a cleanly completed session resets the streak. If every
# port is cooling down, selection falls back to the full set: a 502 is better
# than a dropped connection.
#
# Env:
#   NINFER_ENGINE_PORTS  space-separated engine ports on 127.0.0.1
#   NINFER_LISTEN_PORT   public listen port
#   NINFER_BIND_ADDR     bind address (default 0.0.0.0)

use strict;
use warnings;
use IO::Socket::INET;
use POSIX qw(WNOHANG);

my @ports = split /\s+/, ($ENV{NINFER_ENGINE_PORTS} // '');
@ports or die "NINFER_ENGINE_PORTS is empty\n";
my $listen_port = $ENV{NINFER_LISTEN_PORT} // 8000;
my $bind_addr   = $ENV{NINFER_BIND_ADDR}   // '0.0.0.0';

# Ejection tuning: consecutive refused sessions before a port is excluded, and
# how long it stays out. 30s matches the outer LiteLLM router's cooldown_time.
use constant EJECT_THRESHOLD => 3;
use constant EJECT_COOLDOWN  => 30;

# Live session count per replica, kept in the parent. Children report their exit
# through SIGCHLD, which is when a session is considered finished.
my %live;      # port -> in-flight sessions
my %by_pid;    # child pid -> port
my %refusals;  # port -> consecutive refused sessions
my %cooldown;  # port -> epoch second until which the port is excluded from selection
$live{$_}      = 0 for @ports;
$refusals{$_}  = 0 for @ports;

my $server = IO::Socket::INET->new(
    LocalAddr => $bind_addr,
    LocalPort => $listen_port,
    Listen    => 512,
    Proto     => 'tcp',
    ReuseAddr => 1,
) or die "bind ${bind_addr}:${listen_port}: $!\n";

$SIG{CHLD} = sub {
    while ((my $pid = waitpid(-1, WNOHANG)) > 0) {
        my $port = delete $by_pid{$pid};
        next unless defined $port;
        $live{$port}-- if $live{$port} > 0;
        # Exit 3 = upstream refused (502), 0 = clean session, 1 = internal
        # fork failure. Only the first two speak about the replica.
        my $status = $? >> 8;
        if ($status == 3) {
            $refusals{$port}++;
            $cooldown{$port} = time() + EJECT_COOLDOWN
                if $refusals{$port} >= EJECT_THRESHOLD;
        } elsif ($status == 0) {
            $refusals{$port} = 0;
        }
    }
};

warn "[balancer] listening on ${bind_addr}:${listen_port} -> @ports\n";

while (1) {
    my $client = $server->accept or next;   # EINTR from SIGCHLD is normal

    # Least connections among ports not in cooldown; ties break toward the
    # lower port for determinism. If every port is cooling down, fall back to
    # the full set — a 502 is better than a dropped connection.
    my $now      = time();
    my @eligible = grep { ($cooldown{$_} // 0) <= $now } @ports;
    @eligible    = @ports unless @eligible;
    my ($target) = sort { $live{$a} <=> $live{$b} || $a <=> $b } @eligible;

    # The child blocks on this pipe until the parent has recorded the session
    # (%by_pid/%live), so a fast-exiting child — upstream refused means 502 in
    # microseconds — can never outrun the bookkeeping: the CHLD handler always
    # sees a mapped pid and every increment has a matching decrement.
    my ($sync_rd, $sync_wr);
    pipe($sync_rd, $sync_wr) or do { close $client; next; };

    my $pid = fork;
    if (!defined $pid) { close $client; close $sync_rd; close $sync_wr; next; }

    if ($pid) {                 # parent: account for the session, keep serving
        $by_pid{$pid} = $target;
        $live{$target}++;
        close $sync_rd;
        close $sync_wr;         # unblocks the child's sync read
        close $client;
        next;
    }

    # child: relay this session to the chosen replica
    $SIG{CHLD} = 'DEFAULT';
    close $server;
    close $sync_wr;
    my $sync_buf; read($sync_rd, $sync_buf, 1);   # wait for the parent's record
    close $sync_rd;

    my $upstream = IO::Socket::INET->new(
        PeerAddr => '127.0.0.1',
        PeerPort => $target,
        Proto    => 'tcp',
    );
    unless ($upstream) {
        # Replica refused the connection: answer 502 rather than dropping the
        # client, so the caller sees a real error instead of a reset socket.
        # Exit 3 (not 1): the parent's CHLD handler counts it for ejection.
        print $client "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        close $client;
        exit 3;
    }

    # Two directions, one process each. Unbuffered syswrite preserves SSE token
    # boundaries: a streaming chunk is forwarded the moment it arrives.
    my $down = fork;
    if (!defined $down) { close $client; close $upstream; exit 1; }

    if ($down == 0) {           # upstream -> client (the token stream)
        while (sysread($upstream, my $buf, 65536)) {
            my $off = 0;
            while ($off < length $buf) {
                my $n = syswrite($client, $buf, length($buf) - $off, $off) or last;
                $off += $n;
            }
        }
        shutdown($client, 1);
        exit 0;
    }

    while (sysread($client, my $buf, 65536)) {   # client -> upstream
        my $off = 0;
        while ($off < length $buf) {
            my $n = syswrite($upstream, $buf, length($buf) - $off, $off) or last;
            $off += $n;
        }
    }
    shutdown($upstream, 1);

    waitpid($down, 0);
    close $client;
    close $upstream;
    exit 0;
}