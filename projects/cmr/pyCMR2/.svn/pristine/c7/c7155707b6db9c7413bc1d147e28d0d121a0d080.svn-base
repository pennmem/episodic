import numpy as np
import scipy.io
import math
from glob import glob


def norm_vec(vec):
    """Helper method to normalize a vector"""

    # get the square root of the sum of each element in the dat vector squared
    denom = np.sqrt(np.sum(vec**2))

    # if the resulting denom value equals 0.0, then set this equal to 1.0
    if denom == 0.0:
        return vec
    else:
        # divide each element in the vector by this denom
        return vec/denom


def advance_context(c_in_normed, c_temp, this_beta):
    """Helper function to advance context"""

    # if row vector, force c_in to be a column vector
    if c_in_normed.shape[1] > 1:
        c_in_normed = c_in_normed.T
    assert(c_in_normed.shape[1] == 1)  # sanity check

    # if col vector, force c_temp to be a row vector
    if c_temp.shape[0] > 1:
        c_temp = c_temp.T
    assert(c_temp.shape[0] == 1)  # sanity check

    # calculate rho
    rho = (math.sqrt(1 + (this_beta**2)*
                     ((np.dot(c_temp, c_in_normed)**2) - 1)) -
           this_beta*np.dot(c_temp, c_in_normed))

    # update context
    updated_c = rho*c_temp + this_beta * c_in_normed.T

    # send updated_c out as a col vector
    if updated_c.shape[1] > 1:
        updated_c = updated_c.T

    return updated_c


class CMR2(object):
    """Initialize CMR2 class"""

    def __init__(self, params, nsources, source_info, LSA_mat, data_mat):
        """
        Initialize CMR2 object

        :param params: dictionary containing desired parameter values for CMR2
        :param nsources: If nsources > 0, model will implement source coding
            (See Polyn et al., 2009).

            Note that nsources refers to the number of cells you want to devote
            to source code information, not to the overall number of sources.
            For instance, you might want to represent a single source
            (e.g., emotion) as a vector of multiple source cells.

        :param source_info: matrix containing source-coding information
        :param LSA_mat: matrix containing LSA cos theta values between each item
            in the word pool.
        :param data_mat: matrix containing the lists of items that were
            presented to a given participant. Dividing this up is taken care
            of in the run_CMR2 method.
            You can also divide the data according to session, rather than
            by subject, if desired.  The run_CMR2 method is where you would
            alter this; simply submit sheets of presented items a session
            at a time rather than a subject at a time.

        ndistractors: There are as many distractors as there are lists,
            because presenting a distractor is how we model the shift in context
            that occurs between lists.  Additionally, an initial orthogonal
            item is presented prior to the first list, so that the system does
            not start with context as an empty 0 vector.

            In the weight matrices & context vectors, the distractors' region
            is located after study item indices & before source indices.

        beta_in_play: The update_context_temp() method will always reference
            self.beta_in_play; beta_in_play changes between the
            different beta (context drift) values offered by the
            parameter set, depending on where we are in the simulation.
        """

        # data we are simulating output from
        self.source_info = source_info
        self.LSA_mat = LSA_mat
        self.pres_list_nos = data_mat

        # data structure
        self.nlists = self.pres_list_nos.shape[0]
        self.listlength = self.pres_list_nos.shape[1]

        # n cells in the system to devote to source item info.
        self.nsources = nsources

        # total no. of study items presented to the subject in this session
        self.nstudy_items_presented = self.listlength * self.nlists

        # n cells for distractors in the f, c layers & weight matrices
        self.ndistractors = self.nlists + 1

        # total number of dimensions operating in the system,
        # including all study lists, distractors, and sources
        self.nelements = (self.nstudy_items_presented + self.ndistractors +
                          self.nsources)

        # make a list of all items ever presented to this subject,
        # as well as a sorted version.
        self.all_session_items = np.reshape(self.pres_list_nos,
                                            (self.nlists*self.listlength))
        self.all_session_items_sorted = np.sort(self.all_session_items)

        # set parameters to those input when instantiating the model
        self.params = params

        # beta used by update_context_temp()
        # this beta will be updated to reflect whatever beta is
        # currently indicated for use in updating context.  For example,
        # during encoding, it will be self.params['beta_enc'].  During
        # post-recall, it will be beta_post_rec.  During recall, it will
        # be beta_rec.
        self.beta_in_play = self.params['beta_enc']

        # init leaky accumulator vars
        self.steps = 0

        # track which study item has been presented
        self.study_item_idx = 0
        # track which list has been presented
        self.list_idx = 0

        # track which distractor item has been presented
        self.distractor_idx = self.nstudy_items_presented

        # list of items recalled throughout model run
        self.recalled_items = []

        # set up / initialize weight matrices
        self.M_FC = np.identity(self.nelements) * self.params['scale_fc']
        self.M_CF = np.identity(self.nelements) * self.params['scale_cf']

        # set up / initialize feature and context layers
        self.c_net = np.zeros((self.nelements, 1))
        self.c_old = np.zeros((self.nelements, 1))
        self.f_net = np.zeros((self.nelements, 1))

        # set up & track leaky accumulator & recall vars
        self.x_thresh_full = np.ones(self.nlists * self.listlength)
        self.n_prior_recalls = np.zeros([self.nstudy_items_presented, 1])
        self.nitems_in_race = (self.listlength *
                               self.params['nlists_for_accumulator'])

        # track the list items that have been presented
        self.lists_presented = []

    def clear_system(self):
        """Reset the system to initialized values;
           not used in this code, but implemented in case
           someone wants it later."""

        # track which study item has been presented
        self.study_item_idx = 0
        # track which list has been presented
        self.list_idx = 0
        # track which distractor has been presented
        self.distractor_idx = self.nstudy_items_presented

        # set up / initialize weight matrices
        self.M_FC = np.identity(self.nelements) * self.params['scale_fc']
        self.M_CF = np.identity(self.nelements) * self.params['scale_cf']

        # set up / initialize feature and context layers
        self.c_net = np.zeros((self.nelements, 1))
        self.c_old = np.zeros((self.nelements, 1))
        self.f_net = np.zeros((self.nelements, 1))

        # reset item recall threshold parameters & prior recalled item info
        self.x_thresh_full = np.ones(self.nlists * self.listlength)
        self.n_prior_recalls = np.zeros([self.nstudy_items_presented, 1])

        # track the list items that have been presented
        self.lists_presented = []

    def present_first_list(self):
        """
            Because the weight matrices are off-diagonal empty at the beginning
            of the study, and IF THERE ARE NO REPEATED ITEMS in the first list,
            we can vectorize the operations for the initial list, which will
            save us a little bit of runtime.  We layer on the semantic structure
            *after* this step, to keep this functionality.

            For later lists, we will need to model the presentation of each item
            individually, in case some items repeat and their association
            strengths need to be layered on over the top of their previous
            presentations.

            If you update the model such that you no longer use the fancy
            first-list code (credit to Lynn & Sean!), and instead use the
            operations in the general present_list(), then make sure to remember
            to still layer on the semantic similarity structure before you
            run the initial list.

            thislist_pattern == sample_pattern in the MATLAB code we're
            migrating from.

            For the first list, instead of conducting the matrix operations
            for updating the weight matrices,
            we place orthogonal vec & presented-item values into the appropriate
            M_FC and M_CF indices.

            When these cells are empty, as in the first list (IF the first list
            has no repeated items), then placing the values is the same
            as multiplying, and saves us a little bit of runtime.

            During the simulation, we will track items not according to their
            index of presenation, but rather, according to their location
            in the overall word pool.  This is because if you were to
            ID items simply as "first item presented," "second item presented,"
            etc., you would accidentally treat repeated items as being
            separate items.

            For the first orthogonal item & first-list presentations,
            rho is calculated differently than in later lists.  See
            Lynn & Sean's MATLAB code.

            To speed up item presentation, instead of presenting each item
            individually and running through the matrix operations in
            Lohnas et al. (2015) and other CMR paper equations, we will create a
            triangular matrix containing the beta in play (here beta_enc)
            element-wise multiplied by powers of rho along each sub-diagonal,
            and 0's along the main diagonal.

            This is because if you follow the actual CMR equations for updating
            context, what you end up getting on subsequent item presentations
            are powers of rho * beta. We will then place these items' updates
            to the matrices individually into the matrices.

            The context contributions of the first item to start up the system
            are determined solely by rho.

            At the end of this method, we will update study item index as though
            we have presented the full list. However, we will not update the
            list index until after the recall session has been conducted,
            so that recall takes place with respect to the current list.

        """

        ##########
        #
        # create item-to-word pool map
        #
        ##########

        # get the current list of presented items (the list_idx'th list)
        thislist_pattern = self.pres_list_nos[self.list_idx]

        # get the indices of where this list's words are located
        list1_pattern_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        # track which lists have been presented
        self.lists_presented.append(list1_pattern_indices)

        # initialize rho for the first-presented item
        rho = math.sqrt(1 - self.params['beta_enc']**2)

        ############
        #
        #   Set up M_exp_FC and M_exp_CF matrices
        #
        ############

        M_exp_FC = np.zeros((self.listlength, self.listlength))

        # Get indices of main diagonal
        test_ind = np.diag_indices_from(M_exp_FC)[0]

        # Set off-diagonals equal to ascending powers of rho, beginning rho^0,
        # and then multiply by beta_enc.
        for i in range(self.listlength):
            rows = test_ind[:self.listlength - i - 1]
            cols = test_ind[(i+1):]

            M_exp_FC[rows, cols] = (rho**i) * self.params['beta_enc']

        # For the initial list, M_exp_CF is just the transpose of M_exp_FC.
        M_exp_CF = M_exp_FC.T

        # init. orthogonal item vector;
        # this vector is a vector of the input that the initial orthogonal
        # item makes to every item in the context layer.
        ortho_vec = np.power(np.ones([1, self.listlength]) * rho,
                             range(self.listlength))

        # calculate a matrix of primacy gradients, to implement
        # the primacy effect

        # make a vector containing the desired primacy scalars
        primacy_vec = (self.params['phi_s'] * np.exp(
            -self.params['phi_d'] * np.asarray(range(self.listlength)))
                       + np.ones(self.listlength))

        # make it a matrix to later multiply element-wise onto M_exp_CF
        primacy_mat = np.reshape(np.repeat(
                                primacy_vec.T, repeats=self.listlength, axis=0),
                                newshape=[self.listlength, self.listlength])

        # Scale M_FC and M_CF by their respective gammas
        M_exp_FC = M_exp_FC * self.params['gamma_fc']
        M_exp_CF = M_exp_CF * self.params['gamma_cf']

        # multiply ortho_vector for M_FC by gamma_fc
        ortho_fc = np.multiply(ortho_vec, self.params['gamma_fc'])

        # multiply ortho_vector for M_CF by gamma_cf
        ortho_cf = ortho_vec * self.params['gamma_cf']

        #############
        #
        #   place orthogonal vec & presented-item values into M_FC
        #
        #############

        row_index = self.distractor_idx
        for j in range(len(list1_pattern_indices)):
            col_index = list1_pattern_indices[j]
            self.M_FC[row_index, col_index] += ortho_fc[0, j]

        # place presented-item values into M_FC
        for i in range(len(list1_pattern_indices)):

            # get the row for that list pattern
            row_index = list1_pattern_indices[i]
            for j in range(len(list1_pattern_indices)):
                # get the col for that list pattern
                col_index = list1_pattern_indices[j]

                # place in the value from M_exp_FC
                self.M_FC[row_index, col_index] += M_exp_FC[i, j]

        # scale ortho_vec and M_CF by primacy values
        M_exp_CF = np.multiply(primacy_mat, M_exp_CF)

        ortho_cf = np.multiply(primacy_mat[:, 0], ortho_cf)

        ############
        #
        #   place orthogonal vec & presented-item values into M_CF
        #
        ############

        # column index is the index of the orthogonal item presented.
        # Row indices are the indices of the items presented afterward,
        # with which it is associated.
        col_index = self.distractor_idx
        for j in range(len(list1_pattern_indices)):
            row_index = list1_pattern_indices[j]
            self.M_CF[row_index, col_index] += ortho_cf[0, j]

        # place presented-item values into M_CF

        # column and row indices are the indices of the items being presented
        for i in range(len(list1_pattern_indices)):

            # get the col for that list pattern
            col_index = list1_pattern_indices[i]
            for j in range(len(list1_pattern_indices)):
                # get the row for that list pattern
                row_index = list1_pattern_indices[j]

                # place in M_CF
                self.M_CF[row_index, col_index] += M_exp_CF[j, i]

        ############
        #
        #   update context as though we have presented all items for list1
        #
        ############

        # reverse ortho_vec values and layer into c_net
        for j in range(len(list1_pattern_indices)):
            item_index = list1_pattern_indices[j]
            self.c_net[item_index][0] += (
                self.params['beta_enc'] * ortho_vec[0][self.listlength - j - 1])

        # update context in distractor index location
        self.c_net[self.distractor_idx][0] += rho**self.listlength

        # update the indices tracking our progress through the session
        self.distractor_idx += 1
        self.study_item_idx = self.listlength - 1

    def create_semantic_structure(self):
        """Layer semantic structure onto M_CF (and M_FC, if s_fc is nonzero)

        Dimensions of the LSA matrix for this subject are
        n presented items x n presented items.

        To get item indices, we will subtract 1 from the item ID, since
        item IDs begin at 1, not 0.
        """

        # get all patterns (items) that will be presented to this participant
        all_patterns = self.all_session_items.copy()

        # Sort the items
        self.sorted_patterns = np.sort(all_patterns)

        # Create a mini-LSA matrix with just the items presented to this Subj.
        exp_LSA = np.zeros(
            [self.nstudy_items_presented, self.nstudy_items_presented])

        # Get list-item LSA indices
        for row_idx, this_item in enumerate(self.sorted_patterns):

            # get item's index in the larger LSA matrix
            this_item_idx = this_item - 1

            for col_idx, compare_item in enumerate(self.sorted_patterns):
                # get ID of jth item for LSA cos theta comparison
                compare_item_idx = compare_item - 1

                # get cos theta value between this_item and compare_item
                cos_theta = self.LSA_mat[this_item_idx, compare_item_idx]

                # place into this session's LSA cos theta matrix
                exp_LSA[int(row_idx), int(col_idx)] = cos_theta

        # scale the LSA values by scale_cf and s,
        # as per Lohnas et al., 2015
        cf_exp_LSA = exp_LSA * self.params['s_cf'] * self.params['scale_cf']

        # add the scaled LSA cos theta values onto the appropriate
        # NW quadrant of the M_CF matrix
        self.M_CF[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += cf_exp_LSA

        # scale the LSA values by scale_fc and s_fc,
        # as per Healey et al., 2016
        fc_exp_LSA = exp_LSA * self.params['s_fc'] * self.params['scale_fc']

        # add the scaled LSA cos theta values onto the appropriate
        # NW quadrant of the M_FC matrix
        self.M_FC[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += fc_exp_LSA

    ################################################
    #
    #   Functions defining the recall process
    #
    ################################################

    def leaky_accumulator(self, in_act, noise_vec, x_thresholds):
        """

        :param in_act: Top listlength * 4 item activations
        :param noise_vec: noise values for the accumulator.  These are
            calculated outside this function to save some runtime.
        :param x_thresholds: Threshold each item in the race must reach
            to be recalled. Starts at 1.0 for each item but can increase
            after an item is recalled in order to prevent repetitions.
        :return: Method returns index of the item that won the race,
            the time that elapsed for this item during the process of
            running the accumulator, and the final state of item activations,
            which although not strictly necessary, can be helpful for
            debugging.

        Later, we can build in something here to give people the option of
        letting items never repeat at all.  But for now, we're going to allow
        them to repeat & this can be prevented by appropriate omega & alpha
        parameter values in the params dictionary.

        To facilitate testing changes to this model, you can comment out
        the noise_vec where it is added to x_s in the while loop below.
        """
        # get max number of accumulator cycles for this run of the accumulator
        ncycles = noise_vec.shape[1]
        nitems_in_race  = in_act.shape[0]

        # track whether an item has passed the threshold
        item_has_crossed = False

        # initialize x_s vector with one 0 per accumulating item
        x_s = np.zeros(nitems_in_race)

        # init counter at 1 instead of 0, for consistency with
        # Lynn & Sean's MATLAB code
        cycle_counter = 1

        # initialize list to track items that won the race
        rec_indices = []
        while cycle_counter < ncycles and not item_has_crossed:

            # we don't scale by eta the way Lynn & Sean do, because
            # np.random.normal has already scaled by eta for us.
            # Still have to scale by sq_dt_tau, though.

            x_s = (x_s + (in_act - self.params['kappa']*x_s
                      - (sum(self.params['lamb']*x_s)
                         - (x_s*self.params['lamb'])))
                        * self.params['dt_tau']
                        + noise_vec[:,cycle_counter] * self.params['sq_dt_tau'])

            # In this implementation, accumulating evidence may not be negative.
            x_s[x_s < 0.0] = 0.0

            # if any item's evidence has passed its threshold,
            if np.any(x_s >= x_thresholds):

                # let the system know that an item has crossed;
                # stop accumulating.
                item_has_crossed = True

                # get the indices where items have passed their recall threshold
                rec_indices = np.where(x_s >= x_thresholds)[0]

            cycle_counter += 1

        # calculate elapsed time:
        time = cycle_counter * self.params['dt']

        # if no item has crossed,
        if rec_indices == []:
            winner_index = None

        # else, if an item has crossed:
        else:
            # if more than one item passed the threshold, pick one at random to
            # be the winner
            if len(rec_indices) > 1:
                winner_index = np.random.choice(rec_indices)
            elif len(rec_indices) == 1:
                winner_index = rec_indices[0]

        return winner_index, time, x_s

    ####################
    #
    #   Initialize and run a recall session
    #
    ####################

    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation.

        """

        time_passed = 0
        rec_time_limit = self.params['rec_time_limit']

        nlists_for_accumulator = 4

        # set vars scaling how many items get entered
        # into the leaky accumulator process
        nitems_in_race = self.listlength * nlists_for_accumulator
        nitems_in_session = self.listlength * self.nlists

        # initialize list to store recalled items
        recalled_items = []
        RTs = []
        times_since_start = []

        # run a recall session for the amount of time in rec_time_limit
        while (time_passed < rec_time_limit) \
                and (len(recalled_items) <= self.listlength + 3):

            # get item activations to input to the accumulator
            f_in = np.dot(self.M_CF, self.c_net)

            # sort f_in so we can get the ll * 4 items
            # with the highest activations
            sorted_indices = np.argsort(np.squeeze(f_in[:nitems_in_session]).T)
            sorted_activations = np.sort(f_in[:nitems_in_session].T)

            # get the top-40 activations, and the indices corresponding
            # to their position in the full list of presented items.
            # we need this second value to recover the item's ID later.
            in_activations = sorted_activations[0][
                             (nitems_in_session - nitems_in_race):]
            in_indices = sorted_indices[(nitems_in_session - nitems_in_race):]

            # determine max cycles for the accumulator.
            max_cycles = np.ceil(
                (rec_time_limit - time_passed) / self.params['dt'])

            # Based on model equations, max_cycles should never get down to 0;
            # generally, the accumulator should always run out of time first.
            if max_cycles == 0:
                raise ValueError("max_cycles reached 0!")

            # draw all noise values for the leaky accumulator process
            # ahead of running the accumulator, to save runtime.
            noise = np.random.normal(
                0, self.params['eta'], size=(nitems_in_race, int(max_cycles)))

            # initialize the x_threshold vector
            x_thresh = self.x_thresh_full[in_indices]

            # get the winner of the leaky accumulator, its reaction time,
            # and this race's activation values.
            # x_n isn't strictly necessary, but can be helpful for debugging.
            winner_accum_idx, this_RT, x_n = self.leaky_accumulator(
                in_activations, noise, x_thresh)

            # increment time counter
            time_passed += this_RT

            # If an item was retrieved, recover the item info corresponding
            # to the activation value index retrieved by the accumulator
            if winner_accum_idx is not None:

                # recover item's index from the original pool of item indices
                winner_sorted_idx = in_indices[winner_accum_idx]

                # get original item ID for this item
                winner_ID = np.sort(self.all_session_items)[winner_sorted_idx]

                ##########
                #
                #   Present item & then update system,
                #   as per regular article equations
                #
                ##########

                # reinstantiate this item
                self.present_item(winner_sorted_idx)

            # if no item was retrieved, instantiate a zero-vector
            # in the feature layer
            else:
                self.f_net = np.zeros([1, self.nelements])

            ##########
            #
            #   Whether or not the item is reported for recall,
            #   the item will still update the current context, as below.
            #
            ##########

            self.beta_in_play = self.params['beta_rec']
            self.update_context_temp()

            ############
            #
            #   See if current context is similar enough to the old context;
            #   If not, censor the item
            #
            ############

            # get similarity between c_old and the c retrieved by the item.
            c_similarity = np.dot(self.c_old.T, self.c_net)

            # if sim threshold is passed,
            if (winner_accum_idx is not None) and (
                        c_similarity >= self.params['c_thresh']):

                # store item ID, RT, and time since start of rec session
                recalled_items.append(winner_ID)
                RTs.append(this_RT)
                times_since_start.append(time_passed)

                # Update the item's recall threshold & its prior recalls count
                self.x_thresh_full[winner_sorted_idx] = (
                    1 + self.params['omega'] * (
                        self.params['alpha'] ** self.n_prior_recalls[
                            winner_sorted_idx]))
                self.n_prior_recalls[winner_sorted_idx] += 1
            else:
                continue

        # update counter of what list we're on
        self.list_idx += 1

        return recalled_items, RTs, times_since_start

    def present_item(self, item_idx):
        """Set the f layer to a row vector of 0's with a 1 in the
        presented item location.

        The model code will arrange this as a column vector where
        appropriate."""

        self.f_net = np.zeros([1, self.nelements])
        self.f_net[0][item_idx] = 1

    def update_context_temp(self):
        """Updates the temporal region of the context vector."""
        self.c_old = self.c_net.copy()

        net_cin = np.dot(self.M_FC, self.f_net.T)

        # nelements in temporal subregion
        nelements_temp = self.nstudy_items_presented + self.nlists + 1

        # get region of context that includes all items presented
        # (items, distractors, & orthogonal initial item)
        cin_temp = net_cin[:nelements_temp]

        # norm the temporal region of the c_in vector
        cin_normed = norm_vec(cin_temp)

        # update the temporal region of the network c vector
        net_c_temp = self.c_net[:nelements_temp]
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated

    def present_list(self):
        """
        Method for presenting lists after the first list.

        In CMR2 with
        source coding, this will also need to be used for the first list,
        because the source-coding cells will overlap onto one another,
        and the first-list method of simply sliding values into their
        indices in the weight matrices, without performing matrix operations,
        will no longer be valid.

        Update context using post-recall beta weight if distractor comes
        between lists; use beta_enc if distractor is the first item
        in the system (item serves to initialize context to non-zero values).

        Subjects do not learn the distractor, so we do not update
        the weight matrices following it.

        :return:
        """

        # present distractor prior to this list
        self.present_item(self.distractor_idx)

        # if this is a between-list distractor,
        if self.list_idx > 0:
            self.beta_in_play = self.params['beta_rec_post']

        # else if this is the orthogonal item that starts up the system,
        elif self.list_idx == 0:
            self.beta_in_play = self.params['beta_enc']

        self.update_context_temp()

        # update distractor location for the next list
        self.distractor_idx += 1

        # get vec of primacy values ahead of presenting items,
        # to save some runtime.

        # calculate a vector of primacy gradients
        prim_vec = (self.params['phi_s'] * np.exp(-self.params['phi_d']
                             * np.asarray(range(self.listlength)))
                      + np.ones(self.listlength))

        # get presentation indices for this particular list:
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        # for each item in the current list,
        for i in range(self.listlength):

            #present the item at its appropriate index
            presentation_idx = thislist_pres_indices[i]
            self.present_item(presentation_idx)

            # update the context layer (for now, just the temp region)
            self.beta_in_play = self.params['beta_enc']
            self.update_context_temp()

            # Update the weight matrices

            # Update M_FC
            lrate_fc = self.params['gamma_fc']
            M_FC_exp = np.dot(self.c_old, self.f_net) * lrate_fc
            self.M_FC += M_FC_exp

            # Update M_CF
            lrate_cf = self.params['gamma_cf'] * prim_vec[i]
            M_CF_exp = np.dot(self.f_net.T, self.c_old.T) * lrate_cf
            self.M_CF += M_CF_exp

            # Update location of study item index
            self.study_item_idx += 1


def separate_files(data_path):
    """If data is in one big file, separate out the data into sheets, by subject.

    :param data_path: If using this method, data_path should refer directly
        to a single data file containing the consolidated data across all
        subjects.
    :return: a list of data matrices, separated out by individual subjects.

    Most MATLAB files in CML format keep subject IDs as a vector,
    with the subject ID repeated for each time that a list was presented
    to that subject.
    """

    # will contain stimulus matrices presented to each subject
    Ss_data = []

    # Get list of unique subject IDs
    data_file = scipy.io.loadmat(
        data_path, squeeze_me=True, struct_as_record=False)
    data_pres_list_nos = data_file['data'].pres_itemnos

    # get list of unique subject IDs
    subj_id_map = data_file['data'].subject
    unique_subj_ids = np.unique(subj_id_map)

    # Get locations where each Subj's data starts & stops.
    new_subj_locs = np.unique(
        np.searchsorted(data_file['data'].subject, subj_id_map))

    # Separate data into sets of lists presented to each subject
    for i in range(new_subj_locs.shape[0]):

        # for all but the last list, get the lists that were presented
        # between the first time that subj ID occurs and when the next ID occurs
        if i < new_subj_locs.shape[0] - 1:
            start_lists = new_subj_locs[i]
            end_lists = new_subj_locs[i + 1]

        # once you have reached the last subj, get all the lists from where
        # that ID first occurs until the final list in the dataset
        else:
            start_lists = new_subj_locs[i]
            end_lists = data_pres_list_nos.shape[0]

        # append subject's sheet
        Ss_data.append(data_pres_list_nos[start_lists:end_lists, :])

    return Ss_data, unique_subj_ids


def run_CMR2_singleSubj(data_mat, LSA_mat, params, source_info):

    """Run CMR2 for an individual subject / data sheet"""

    # init. lists to store CMR2 output
    resp_values = []
    RT_values = []
    time_values = []

    # create CMR2 object
    this_CMR = CMR2(
        params=params, nsources=0, source_info=[],
        LSA_mat=LSA_mat, data_mat=data_mat)

    # Present first list.  Different algorithm than later lists;
    # see method documentation.
    this_CMR.present_first_list()

    # layer LSA cos theta values onto the weight matrices
    this_CMR.create_semantic_structure()

    # Recall the first list
    rec_items, RTs_thislist, times_from_start \
        = this_CMR.recall_session()

    # Format output in zero-padded matrix
    resp_values.append(rec_items)
    RT_values.append(RTs_thislist)
    time_values.append(times_from_start)

    # Run CMR2 for all lists after the 0th list
    for i in range(len(this_CMR.pres_list_nos) - 1):
        # present new list
        this_CMR.present_list()

        # recall session
        rec_items_i, RTs_list_i, times_from_start_i \
            = this_CMR.recall_session()

        # append recall responses & times
        resp_values.append(rec_items_i)
        RT_values.append(RTs_list_i)
        time_values.append(times_from_start_i)

    return resp_values, RT_values, time_values


def run_CMR2(LSA_path, data_path, params, sep_files, filename_stem=""):
    """Run CMR2 for all subjects

    time_values = time for each item since beginning of recall session

    For later zero-padding the output, we will get list length from the
    width of presented-items matrix. This assumes equal list lengths
    across Ss and sessions, unless you are inputting each session
    individually as its own matrix, in which case, list length will
    update accordingly.

    If all Subjects' data are combined into one big file, as in some files
    from prior CMR2 papers, then divide data into individual sheets per subj.

    If you want to simulate CMR2 for individual sessions, then you can
    feed in individual session sheets at a time, rather than full subject
    presented-item sheets.
    """

    # load in LSA information & format for CMR (populate main diag. with 0's)
    LSA_file = scipy.io.loadmat(
        LSA_path, squeeze_me=True, struct_as_record=False)
    LSA_mat = LSA_file['LSA'].copy()
    np.fill_diagonal(LSA_mat, 0)

    # init. lists to store CMR2 output
    resp_vals_allSs = []
    RT_vals_allSs = []
    time_vals_allSs = []

    # Simulate each subject's responses.
    if not sep_files:

        # divide up the data
        subj_presented_data, unique_subj_ids = separate_files(data_path)

        # get list length
        listlength = subj_presented_data[0].shape[1]

        # for each subject's data matrix,
        for m, data_mat in enumerate(subj_presented_data):

            subj_id = unique_subj_ids[m]
            print('Subject ID is: ' + str(subj_id))

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                data_mat=data_mat, LSA_mat=LSA_mat,
                params=params, source_info=[])

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

    # If files are separate, then read in each file individually
    # so that we don't end up having to make & maintain a potentially
    # massive data matrix.
    else:

        # get all the individual data file paths
        indiv_file_paths = glob(data_path + filename_stem + "*.mat")

        # read in the data for each path & stick it in a list of data matrices
        for file_path in indiv_file_paths:

            data_file = scipy.io.loadmat(
                file_path, squeeze_me=True, struct_as_record=False)  # get data
            data_mat = data_file['data'].pres_itemnos  # get presented items

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                data_mat=data_mat, LSA_mat=LSA_mat,
                params=params, source_info=[])

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

        # for later zero-padding the output, get list length from one file.
        data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                     struct_as_record=False)
        data_mat = data_file['data'].pres_itemnos

        listlength = data_mat.shape[1]


    ##############
    #
    #   Zero-pad the output
    #
    ##############

    # If more than one subject, reshape the output into a single,
    # consolidated sheet across all Ss
    if len(resp_vals_allSs) > 0:
        resp_values = [item for submat in resp_vals_allSs for item in submat]
        RT_values = [item for submat in RT_vals_allSs for item in submat]
        time_values = [item for submat in time_vals_allSs for item in submat]
    else:
        resp_values = resp_vals_allSs
        RT_values = RT_vals_allSs
        time_values = time_vals_allSs

    # set max width for zero-padded response matrix
    maxlen = listlength * 2

    nlists = len(resp_values)

    # init. zero matrices of desired shape
    resp_mat  = np.zeros((nlists, maxlen))
    RTs_mat   = np.zeros((nlists, maxlen))
    times_mat = np.zeros((nlists, maxlen))

    # place output in from the left
    for row_idx, row in enumerate(resp_values):

        resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
        RTs_mat[row_idx][:len(row)]   = RT_values[row_idx]
        times_mat[row_idx][:len(row)] = time_values[row_idx]


    print('Analyses complete.')

    return resp_mat, times_mat


def main():
    """Main method"""

    # set desired parameters. Example below is for Kahana et al. (2002),
    # from Lohnas et al. (2015)
    params_K02 = {

        'beta_enc': 0.519769,
        'beta_rec': 0.627801,
        'gamma_fc': 0.425064,
        'gamma_cf': 0.895261,
        'scale_fc': 1 - 0.425064,
        'scale_cf': 1 - 0.895261,

        'phi_s': 1.408899,
        'phi_d': 0.989567,
        'kappa': 0.312686,

        'eta': 0.392847,
        's_cf': 1.292411,
        's_fc': 0.0,
        'beta_rec_post': 0.802543,
        'omega': 11.894106,
        'alpha': 0.678955,
        'c_thresh': 0.073708,
        'dt': 10.0,

        'lamb': 0.129620,
        'rec_time_limit': 30000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 4
    }

    # format printing nicely
    np.set_printoptions(precision=5)

    # Set LSA and data paths -- K02 data
    LSA_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_LSA.mat'
    data_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_data.mat'

    rec_nos, times = run_CMR2(LSA_path, data_path, params_K02, sep_files=False)

    np.savetxt('resp_K02.txt', np.asmatrix(rec_nos), delimiter=',', fmt='%.0d')
    np.savetxt('times_K02.txt', np.asmatrix(times), delimiter=',', fmt='%.0d')

if __name__ == "__main__": main()
