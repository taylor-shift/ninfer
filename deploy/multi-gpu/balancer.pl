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

# Live session count per replica, kept in the parent. Children report their exit
# through SIGCHLD, which is when a session is considered finished.
my %live;      # port -> in-flight sessions
my %by_pid;    # child pid -> port
$live{$_} = 0 for @ports;

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
        $live{$port}-- if defined $port && $live{$port} > 0;
    }
};

warn "[balancer] listening on ${bind_addr}:${listen_port} -> @ports\n";

while (1) {
    my $client = $server->accept or next;   # EINTR from SIGCHLD is normal

    # Least connections; ties break toward the lower port for determinism.
    my ($target) = sort { $live{$a} <=> $live{$b} || $a <=> $b } @ports;

    my $pid = fork;
    if (!defined $pid) { close $client; next; }

    if ($pid) {                 # parent: account for the session, keep serving
        $by_pid{$pid} = $target;
        $live{$target}++;
        close $client;
        next;
    }

    # child: relay this session to the chosen replica
    $SIG{CHLD} = 'DEFAULT';
    close $server;

    my $upstream = IO::Socket::INET->new(
        PeerAddr => '127.0.0.1',
        PeerPort => $target,
        Proto    => 'tcp',
    );
    unless ($upstream) {
        # Replica refused the connection: answer 502 rather than dropping the
        # client, so the caller sees a real error instead of a reset socket.
        print $client "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        close $client;
        exit 1;
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
