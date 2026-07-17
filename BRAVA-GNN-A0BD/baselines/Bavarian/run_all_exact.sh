#! /bin/sh
mkdir -p exact/ && \
./bin/exact -jfs data/email-Enron.dat > exact/email-Enron.json && \
./bin/exact -jfs data/ca-AstroPH.dat > exact/ca-AstroPH.json && \
./bin/exact -jfs data/cit-HepPh.dat > exact/cit-HepPh.json && \
./bin/exact -jfs data/p2p-Gnutella31.dat > exact/p2p-Gnutella31.json && \
./bin/exact -jfs data/soc-Epinions1.dat > exact/soc-Epinions1.json && \
./bin/exact -jfs data/loc-gowalla.dat > exact/loc-gowalla.json && \
./bin/exact -jfs data/com-dblp.dat > exact/com-dblp.json
# We do not run the exact algorithm for com-youtube and as-skitter because it would take too long.
