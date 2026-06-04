import streamlit as st
import chess
import chess.svg
import torch
import pandas as pd
import sys
import os

# Add project root to path for imports
sys.path.append(os.getcwd())

from engine.alphabeta import AlphaBetaEngine
from engine.mcts import MCTSEngine
from engine.encoder import BoardEncoder
from neural.model import ChessResNet

# --- Setup ---
st.set_page_config(page_title="ChessRL Dashboard", layout="wide")
st.title("♟️ ChessRL: Neural vs Classical Engine")

# --- State Management ---
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()

if 'engine_type' not in st.session_state:
    st.session_state.engine_type = "Classical (Alpha-Beta)"

if 'last_move' not in st.session_state:
    st.session_state.last_move = None

# --- Sidebar: Engine Config & Telemetry ---
st.sidebar.header("Engine Configuration")
engine_option = st.sidebar.selectbox(
    "Select Opponent:",
    ("Classical (Alpha-Beta)", "Neural (MCTS)")
)
st.session_state.engine_type = engine_option

st.sidebar.divider()
st.sidebar.header("Live Telemetry")

nodes_placeholder = st.sidebar.empty()
eval_placeholder = st.sidebar.empty()

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    # Render Board with highlighting
    last_move = st.session_state.board.peek() if st.session_state.board.move_stack else None
    board_svg = chess.svg.board(
        board=st.session_state.board,
        lastmove=last_move,
        size=600
    )
    st.write(board_svg, unsafe_allow_html=True)

with col2:
    st.header("Play Move")
    
    # Text input for UCI move
    move_input = st.text_input("Enter UCI Move (e.g. e2e4):", placeholder="e2e4")
    
    if st.button("Submit Move") or (move_input and len(move_input) == 4):
        try:
            move = chess.Move.from_uci(move_input.lower())
            if move in st.session_state.board.legal_moves:
                st.session_state.board.push(move)
                st.session_state.last_move = move
                
                # Engine's Turn
                if not st.session_state.board.is_game_over():
                    with st.spinner("Engine is thinking..."):
                        try:
                            # IMPORTANT: Always use a copy of the board for engines to prevent state corruption
                            board_copy = st.session_state.board.copy()
                            
                            if st.session_state.engine_type == "Classical (Alpha-Beta)":
                                engine = AlphaBetaEngine()
                                # Classical search on a copy
                                best_move = engine.search(board_copy, max_depth=5)
                                nodes_placeholder.metric("Nodes Evaluated", engine.nodes_visited)
                            else:
                                model = ChessResNet()
                                encoder = BoardEncoder()
                                model.load_state_dict(torch.load("models/supervised_base.pt", map_location='cpu'))
                                engine = MCTSEngine(model, encoder)
                                # Boost iterations to 1000 for higher Elo
                                best_move = engine.search(board_copy, iterations=1000)
                                nodes_placeholder.metric("MCTS Simulations", 1000)
                            
                            if best_move and best_move in st.session_state.board.legal_moves:
                                st.session_state.board.push(best_move)
                            else:
                                st.error("Engine returned an illegal move or failed.")
                        except Exception as e:
                            st.error(f"Engine Error: {e}")
                
                # Clear input and refresh
                st.rerun()
            else:
                st.error("Illegal Move!")
        except ValueError:
            if len(move_input) >= 4:
                st.error("Invalid UCI format!")

    st.divider()
    st.header("Controls")
    if st.button("Reset Game"):
        st.session_state.board.reset()
        st.session_state.last_move = None
        st.rerun()
    if st.button("Undo Move"):
        if st.session_state.board.move_stack:
            st.session_state.board.pop()
            if st.session_state.board.move_stack:
                st.session_state.board.pop()
            st.rerun()

# --- Game Status ---
if st.session_state.board.is_game_over():
    res = st.session_state.board.result()
    st.success(f"Game Over! Result: {res}")
