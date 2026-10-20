#!/usr/bin/env python3
"""Independent fixtures and production-object regressions for the two fixes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import chess

HERE = Path(__file__).resolve().parent
CHECKS = HERE.parent / 'output/regressions'
CHECKS.mkdir(parents=True, exist_ok=True)
SOURCE = HERE.parent / 'engine'
BUILD = HERE.parent / 'build'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror_move(uci):
    m = chess.Move.from_uci(uci)
    return chess.Move(chess.square_mirror(m.from_square),
                      chess.square_mirror(m.to_square), m.promotion).uci()


def identity(board):
    return (board.board_fen(), board.turn, board.clean_castling_rights(),
            board.ep_square if board.has_legal_en_passant() else None)


def game_result(board):
    if board.is_checkmate():
        return 3 if board.turn == chess.BLACK else 1
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.halfmove_clock >= 100 or board.is_repetition(3)):
        return 2
    return 0


def cases():
    old = json.loads((HERE / 'fixtures/witnesses.json').read_text())
    histories = [{'id': f['name'], 'fen': f['initial_fen'], 'moves': f['moves']}
                 for f in old['fixtures']]
    cycle = ['g1f3', 'g8h8', 'f3g1', 'h8g8']
    extra = [
        ('pinned_loaded_fen', '4r1k1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 2', cycle * 3),
        ('zeroing_after_repetition', '4r1k1/3p4/8/4P3/8/8/8/4K1N1 b - - 0 1',
         ['d7d5'] + cycle * 2 + ['e5e6']),
        ('castling_right_lost', '4k1n1/8/8/8/8/8/8/R3K1NR w KQ - 0 1',
         ['h1h2','g8f6','h2h1','f6g8'] * 2),
        ('castling_right_kept', '4k1n1/8/8/8/8/8/8/R3K1NR w KQ - 0 1',
         ['g1f3','g8f6','f3g1','f6g8'] * 2),
    ]
    for name, fen, moves in extra:
        histories.append({'id':name, 'fen':fen, 'moves':moves})
        histories.append({'id':name+'_mirrored',
                          'fen':chess.Board(fen).mirror().fen(en_passant='fen'),
                          'moves':[mirror_move(m) for m in moves]})

    ep_positions = [
        ('vertical_pin', '4r1k1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 2'),
        ('horizontal_pin', '7k/8/8/r4pPK/8/8/8/8 w - f6 0 1'),
        ('diagonal_pin', '7k/8/5b2/3pP3/3K4/8/8/8 w - d6 0 1'),
        ('capture_checking_pawn', '7k/8/8/3pP3/4K3/8/8/8 w - d6 0 2'),
        ('unrelated_discovered_check', '2b4k/8/8/3pP3/8/7K/8/8 w - d6 0 1'),
        ('left_capturer_free', '4r2k/8/8/2PpP3/8/8/8/4K3 w - d6 0 1'),
        ('right_capturer_free', '2r4k/8/8/2PpP3/8/8/8/2K5 w - d6 0 1'),
        ('adjacent_nonpawns', '7k/8/8/2NpN3/8/8/8/K7 w - d6 0 1'),
        ('edge_file_a', '7k/8/8/pP6/8/8/8/7K w - a6 0 1'),
        ('edge_file_h', 'k7/8/8/6Pp/8/8/8/K7 w - h6 0 1'),
        ('no_adjacent_pawn', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'),
    ]
    pairs = []
    def add_pair(name, a, b):
        require(a.is_valid() and b.is_valid(), 'invalid pair fixture: '+name)
        pairs.append({'id':name, 'first':a.fen(en_passant='fen'),
                      'second':b.fen(en_passant='fen'),
                      'same_repetition':identity(a)==identity(b),
                      'same_exact':a.fen(en_passant='fen').split()[:4] == b.fen(en_passant='fen').split()[:4],
                      'legal_ep_first':[m.uci() for m in a.generate_legal_ep()],
                      'legal_ep_second':[m.uci() for m in b.generate_legal_ep()]})
    for name, fen in ep_positions:
        for mirrored in (False,True):
            a = chess.Board(fen)
            if mirrored:
                a = a.mirror()
            name2 = name + ('_mirrored' if mirrored else '')
            b = a.copy(); b.ep_square = None
            add_pair(name2, a, b)
            add_pair(name2+'_identical', a, a)
    for name, fen in [
        ('two_legal_targets','7k/8/8/3pPp2/8/8/8/K7 w - d6 0 1'),
        ('two_illegal_targets','7k/8/8/3p1p2/8/8/8/K7 w - d6 0 1'),
        ('one_legal_target','7k/8/8/2Pp1p2/8/8/8/K7 w - d6 0 1'),
    ]:
        a = chess.Board(fen); b = a.copy(); b.ep_square = chess.F6
        add_pair(name,a,b); add_pair(name+'_mirrored',a.mirror(),b.mirror())
    a = chess.Board()
    b = a.copy(); b.turn = chess.BLACK
    add_pair('side_to_move_differs',a,b)
    b = a.copy(); b.castling_rights &= ~chess.BB_H1
    add_pair('castling_rights_differ',a,b)
    b = a.copy(); b.remove_piece_at(chess.A2)
    add_pair('piece_placement_differs',a,b)
    b = a.copy(); b.halfmove_clock = 12; b.fullmove_number = 40
    add_pair('counters_do_not_change_board_identity',a,b)

    expected = []
    for case in histories:
        board = chess.Board(case['fen'])
        require(board.is_valid(), 'invalid history start '+case['id'])
        keys = []
        for ply in range(len(case['moves'])+1):
            if ply:
                move = chess.Move.from_uci(case['moves'][ply-1])
                require(move in board.legal_moves, 'illegal reference move '+case['id'])
                board.push(move)
            key = identity(board)
            previous = [i for i,k in enumerate(keys) if k==key]
            # Derive nearest-cycle length by board identity, independently of
            # the engine's scan/cached repetition algorithm.
            count = max(n-1 for n in range(1,10) if board.is_repetition(n))
            require(count == len(previous), 'reference count disagreement')
            expected.append({'id':case['id'], 'ply':ply, 'repetitions':count,
                             'cycle':ply-previous[-1] if previous else 0,
                             'game_result':game_result(board),
                             'fen':board.fen(en_passant='fen' if ply==0 else 'xfen')})
            keys.append(key)
    return histories, pairs, expected


def compile_harness():
    text = (SOURCE/'src/search/classic/search.cc').read_text()
    start = text.index('[draw_score]',text.index('std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature'))
    stop = text.index('\n      });',start)
    comparator = text[start:stop]+'\n      }'
    require('If both are terminal draws' not in comparator, 'old comparator remains')
    body = (HERE/'audit_checks.cc.in').read_text().replace('__EXTRACTED_COMPARATOR__',comparator)
    (CHECKS/'audit_checks.cc').write_text(body)
    commands = json.loads((BUILD/'compile_commands.json').read_text())
    entry = next(x for x in commands if x['file'].endswith('/src/search/classic/node.cc'))
    args = shlex.split(entry['command'])
    for flag in ['-o','-MF','-MQ']:
        if flag in args:
            i = args.index(flag); del args[i:i+2]
    args = [x for x in args if x not in ['-MD','-MMD','-c'] and
            not x.endswith('/src/search/classic/node.cc')]
    obj = CHECKS/'audit_checks.o'
    args += ['-UNDEBUG','-c',str(CHECKS/'audit_checks.cc'),'-o',str(obj)]
    link = shlex.split(subprocess.check_output(
        ['ninja','-t','commands','lc0'],cwd=BUILD,text=True).splitlines()[-1])
    link = [x for x in link if not x.endswith('src_main.cc.o')]
    link[link.index('-o')+1] = str(CHECKS/'audit_checks')
    link.append(str(obj))
    (CHECKS/'audit-build.json').write_text(json.dumps({
        'compile':args,'link':link,
        'comparator_sha256':hashlib.sha256(comparator.encode()).hexdigest(),
        'production_search_sha256':digest(SOURCE/'src/search/classic/search.cc')},indent=2)+'\n')
    subprocess.run(args,cwd=BUILD,check=True)
    subprocess.run(link,cwd=BUILD,check=True)


def run():
    histories, pairs, expected = cases()
    fixture = {'histories':histories,'pairs':pairs,'expected_histories':expected,
               'reference':'python-chess '+chess.__version__}
    (CHECKS/'audit-cases.json').write_text(json.dumps(fixture,indent=2)+'\n')
    compile_harness()
    requests = [f"history\t{h['id']}\t{h['fen']}\t{' '.join(h['moves'])}\n" for h in histories]
    requests += [f"pair\t{p['id']}\t{p['first']}\t{p['second']}\n" for p in pairs]
    request = ''.join(requests)
    (CHECKS/'audit.stdin').write_text(request)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    proc = subprocess.run([str(CHECKS/'audit_checks')],input=request,text=True,
                          capture_output=True,timeout=8,env=env)
    (CHECKS/'audit.stdout').write_text(proc.stdout)
    (CHECKS/'audit.stderr').write_text(proc.stderr)
    require(proc.returncode==0,'audit regression failed: '+proc.stderr[-2000:])
    rows = [line.split('\t') for line in proc.stdout.splitlines()]
    actual_history = [row for row in rows if row[0]=='history']
    actual_pairs = [row for row in rows if row[0]=='pair']
    require(len(actual_history)==len(expected),'missing history results')
    require(len(actual_pairs)==len(pairs),'missing identity pairs')
    for row,want in zip(actual_history,expected):
        require(row[1]==want['id'] and int(row[2])==want['ply'],'history ordering mismatch')
        for idx,key in [(3,'repetitions'),(4,'cycle'),(5,'game_result')]:
            require(int(row[idx])==want[key],f"{want['id']} ply {want['ply']}: {key} {row[idx]} != {want[key]}")
        require(row[6]==want['fen'],'raw history FEN changed: '+want['id'])
    for row,want in zip(actual_pairs,pairs):
        require(row[1]==want['id'],'pair ordering mismatch')
        require(bool(int(row[2]))==want['same_exact'],'exact equality changed: '+want['id'])
        require(bool(int(row[3]))==want['same_repetition'],'repetition identity mismatch: '+want['id'])
        require(bool(int(row[4]))==want['same_exact'],'exact hash distinction changed: '+want['id'])
        require(row[5]==want['first'] and row[6]==want['second'],'raw pair FEN changed: '+want['id'])
    comparator = next(row for row in rows if row[0]=='comparator')
    checks = next(row for row in rows if row[0]=='checks')
    result = {'status':'passed', 'histories':len(histories),
              'history_prefixes':len(expected),'identity_pairs':len(pairs),
              'comparator_states':int(comparator[2]),'draw_scores':[-1,0,1],
              'cpp_checks':int(checks[1]),'reference':fixture['reference'],
              'gpu':False,'engine_search':False,'node_bytes':64,
              'harness_sha256':digest(CHECKS/'audit_checks')}
    (CHECKS/'audit-results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
